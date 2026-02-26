import { callLLMStream, callLLMStreamTokens } from "../services/llm.service.js";
import { classifyIntent } from "./intent.js";
import { queryBySQL } from "./sql.js";
import {
  SYSTEM_SQL_STATS,
  SYSTEM_SQL_CASE,
  SYSTEM_SMALLTALK,
  SYSTEM_GENERAL,
} from "./prompts.js";

// =========================
// 헬퍼: 이모지 제거
// =========================
function stripEmojis(str) {
  return str
    .replace(/\p{Extended_Pictographic}/gu, "")
    .replace(/[\u200D\uFE0F\uFE0E]/g, "")
    .replace(/[ \t]{2,}/g, " ");
}

// =========================
// 헬퍼: 오늘 날짜 문자열
// =========================
function getTodayPrefix() {
  const now = new Date();
  const days = ["일", "월", "화", "수", "목", "금", "토"];
  return `오늘 날짜: ${now.getFullYear()}년 ${now.getMonth() + 1}월 ${now.getDate()}일 (${days[now.getDay()]}요일).`;
}

// =========================
// 핵심: 인텐트 분류 → SQL 조회 → 컨텍스트 결정
// =========================
async function _searchAndBuild(rawQuestion) {
  const { intent, query: question, entities } = await classifyIntent(rawQuestion);

  let systemPrompt = SYSTEM_GENERAL;
  let context = "";
  let rag_mode = "general";
  const sources = [];

  if (intent === "smalltalk") {
    systemPrompt = SYSTEM_SMALLTALK;
    rag_mode = "smalltalk";
  } else {
    try {
      const { context: sqlCtx } = await queryBySQL(question, intent, entities);
      if (sqlCtx) {
        context = sqlCtx;
        rag_mode = intent === "stats" ? "sql-stats" : "sql-case";
        systemPrompt = intent === "stats" ? SYSTEM_SQL_STATS : SYSTEM_SQL_CASE;
      } else {
        // SQL 성공했으나 결과 없음
        rag_mode = intent === "stats" ? "no-data-stats" : "no-data-case";
      }
    } catch (e) {
      console.warn("[chain] SQL 실패, general 모드 전환:", e.message);
      rag_mode = "general";
    }
  }

  systemPrompt = `${getTodayPrefix()}\n\n${systemPrompt}`;
  return { question, intent, entities, systemPrompt, context, sources, rag_mode };
}

// =========================
// 헬퍼: userContent 생성
// =========================
function buildUserContent(question, context, rag_mode, entities) {
  if (!context) {
    if (rag_mode === "no-data-stats" && entities?.date_ref) {
      return `[시스템 안내] 사용자가 요청한 날짜(${entities.date_ref})의 화재 통계 데이터가 데이터베이스에 없습니다. 이 사실을 먼저 명확히 알리고, 통계 조회 방법이나 대략적인 일반 정보를 안내하세요.\n\n사용자 질문: ${question}`;
    }
    if (rag_mode === "no-data-case") {
      return `[시스템 안내] 해당 조건의 화재 사례 데이터가 데이터베이스에 없습니다. 이 사실을 명확히 알리고, 전문 지식을 바탕으로 답변하세요.\n\n사용자 질문: ${question}`;
    }
    return question;
  }

  return `아래는 사용자 질문과 관련된 데이터베이스 조회 결과입니다. 이 데이터가 바로 사용자가 찾는 정보입니다. 분류명(예: 창고시설, 공동주택)이 사용자 표현(예: 물류센터, 아파트)과 달라도 동일한 데이터로 취급하고 반드시 답변에 활용하세요.\n\n<db_result>\n${context}\n</db_result>\n\n사용자 질문: ${question}`;
}

// =========================
// 헬퍼: 메시지 배열 생성
// =========================
function buildMessages(systemPrompt, history, userContent) {
  const messages = [{ role: "system", content: systemPrompt }];
  if (Array.isArray(history)) {
    history.forEach((turn) => {
      if (turn.user)      messages.push({ role: "user",      content: turn.user });
      if (turn.assistant) messages.push({ role: "assistant", content: turn.assistant });
    });
  }
  messages.push({ role: "user", content: userContent });
  return messages;
}

// =========================
// 공개 API: non-streaming
// =========================
export async function runRag({
  question,
  history = [],
  max_new_tokens = 800,
  temperature = 0.3,
  top_p = 0.9,
}) {
  const { question: q, entities, systemPrompt, context, sources, rag_mode } =
    await _searchAndBuild(question);

  const messages = buildMessages(systemPrompt, history, buildUserContent(q, context, rag_mode, entities));
  // 통계 모드는 일관된 수치 답변을 위해 낮은 temperature 강제
  const effectiveTemp = rag_mode === "sql-stats" ? Math.min(temperature, 0.1) : temperature;
  const txt = await callLLMStream(messages, { maxTokens: max_new_tokens, temperature: effectiveTemp, top_p });

  return { answer: stripEmojis(txt), sources, rag_mode };
}

// =========================
// 공개 API: SSE 스트리밍
// =========================
export async function runRagStream({
  question,
  history = [],
  max_new_tokens = 800,
  temperature = 0.3,
  top_p = 0.9,
  onToken,
}) {
  const { question: q, entities, systemPrompt, context, sources, rag_mode } =
    await _searchAndBuild(question);

  const messages = buildMessages(systemPrompt, history, buildUserContent(q, context, rag_mode, entities));
  const effectiveTemp = rag_mode === "sql-stats" ? Math.min(temperature, 0.1) : temperature;
  await callLLMStreamTokens(
    messages,
    { maxTokens: max_new_tokens, temperature: effectiveTemp, top_p },
    (token) => onToken(stripEmojis(token)),
  );

  return { sources, rag_mode };
}

export default { runRag, runRagStream };
