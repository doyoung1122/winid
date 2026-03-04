import { callLLMStream, callLLMStreamTokens } from "../services/llm.service.js";
import { classifyIntent } from "./intent.js";
import { queryBySQL } from "./sql.js";
import { searchByQuery, buildContext } from "../services/vector.service.js";
import { USE_AS_CTX_MIN } from "../config/env.js";
import {
  SYSTEM_SQL_STATS,
  SYSTEM_SQL_CASE,
  SYSTEM_SMALLTALK,
  SYSTEM_GENERAL,
  SYSTEM_RAG_DOC,
} from "./prompts.js";

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
async function _searchAndBuild(rawQuestion, history = []) {
  const { intent, query: question, entities } = await classifyIntent(rawQuestion, history);

  let systemPrompt = SYSTEM_GENERAL;
  let context = "";
  let rag_mode = "general";
  const sources = [];

  if (intent === "smalltalk") {
    systemPrompt = SYSTEM_SMALLTALK;
    rag_mode = "smalltalk";

  } else if (intent === "stats") {
    // 통계: SQL만 조회
    const sqlResult = await queryBySQL(question, intent, entities).catch(e => {
      console.warn("[chain] SQL 실패:", e.message);
      return { context: null };
    });
    const rawSqlCtx = sqlResult?.context || null;
    const sqlCtx = rawSqlCtx && rawSqlCtx.length > 4000
      ? rawSqlCtx.slice(0, 4000) + "\n... (이하 생략)"
      : rawSqlCtx;
    if (sqlCtx) {
      context = sqlCtx;
      rag_mode = "sql-stats";
      systemPrompt = SYSTEM_SQL_STATS;
    } else {
      rag_mode = "no-data-stats";
    }

  } else if (intent === "document") {
    // 문서/지식: ChromaDB만 조회 (SQL 불필요)
    // 이중 검색: 원문(한국어) + 번역(영어) 병렬 → 결과 병합
    // 한글 문서와 영문 문서(NFPA 등) 모두 커버
    let translated = null;
    if (/[가-힣]/.test(question)) {
      try {
        const t = await callLLMStream(
          [
            { role: "system", content: "Translate the Korean fire investigation question to English. Output only the English translation, nothing else." },
            { role: "user", content: question },
          ],
          { maxTokens: 150, temperature: 0.1 }
        );
        if (t && t.trim().length > 5) {
          translated = t.trim();
          console.log(`[chain] KO→EN: "${translated}"`);
        }
      } catch (e) {
        console.warn("[chain] 번역 실패:", e.message);
      }
    }

    // 원문 + 번역 병렬 검색
    const searchTasks = [
      searchByQuery(question).catch(() => ({ matches: [], maxSim: 0 })),
      ...(translated ? [searchByQuery(translated).catch(() => ({ matches: [], maxSim: 0 }))] : []),
    ];
    const searchResults = await Promise.all(searchTasks);

    // 결과 병합: ID 중복 제거 후 유사도 내림차순
    const seen = new Set();
    const merged = [];
    for (const r of searchResults) {
      for (const m of r.matches || []) {
        const key = m.id ?? m.content?.slice(0, 80);
        if (!seen.has(key)) { seen.add(key); merged.push(m); }
      }
    }
    merged.sort((a, b) => b.similarity - a.similarity);
    const docMatches = merged.slice(0, 5);
    const docMaxSim = docMatches.length > 0 ? docMatches[0].similarity : 0;

    const hasDoc = docMaxSim >= USE_AS_CTX_MIN && docMatches.length > 0;
    if (hasDoc) {
      const { context: docCtx, sources: docSources } = buildContext(docMatches, 2500);
      if (docCtx) {
        context = docCtx;
        sources.push(...docSources);
        rag_mode = "rag-doc";
        systemPrompt = SYSTEM_RAG_DOC;
      }
    }
    if (!context) rag_mode = "no-data-document";

  } else {
    // case: SQL + ChromaDB 병렬 조회
    const [sqlResult, chromaResult] = await Promise.allSettled([
      queryBySQL(question, intent, entities).catch(e => {
        console.warn("[chain] SQL 실패:", e.message);
        return { context: null };
      }),
      searchByQuery(question).catch(e => {
        console.warn("[chain] ChromaDB 실패:", e.message);
        return { matches: [], maxSim: 0 };
      }),
    ]);

    const rawSqlCtx =
      sqlResult.status === "fulfilled" ? sqlResult.value?.context || null : null;
    const sqlCtx = rawSqlCtx && rawSqlCtx.length > 4000
      ? rawSqlCtx.slice(0, 4000) + "\n... (이하 생략)"
      : rawSqlCtx;
    const { matches: docMatches, maxSim: docMaxSim } =
      chromaResult.status === "fulfilled"
        ? chromaResult.value
        : { matches: [], maxSim: 0 };
    const hasDoc = docMaxSim >= USE_AS_CTX_MIN && docMatches.length > 0;

    // case: SQL + 업로드 문서 조합
    if (sqlCtx && hasDoc) {
      const { context: docCtx, sources: docSources } = buildContext(docMatches, 1500);
      context = sqlCtx + (docCtx ? `\n\n---\n\n[참조 문서]\n${docCtx}` : "");
      sources.push(...docSources);
      rag_mode = "sql-case";
      systemPrompt = SYSTEM_SQL_CASE;
    } else if (sqlCtx) {
      context = sqlCtx;
      rag_mode = "sql-case";
      systemPrompt = SYSTEM_SQL_CASE;
    } else if (hasDoc) {
      const { context: docCtx, sources: docSources } = buildContext(docMatches, 2000);
      if (docCtx) {
        context = docCtx;
        sources.push(...docSources);
        rag_mode = "rag-doc";
        systemPrompt = SYSTEM_RAG_DOC;
      }
    }

    if (!context) rag_mode = "no-data-case";
  }

  systemPrompt = `${getTodayPrefix()}\n\n${systemPrompt}`;
  return { question, intent, entities, systemPrompt, context, sources, rag_mode };
}

// =========================
// 헬퍼: 질문·엔티티에서 조사 맥락 노트 생성 (코드 기반, LLM 불필요)
// =========================
function buildContextNote(question, entities) {
  const lines = [];

  // 1) 건물 유형 → 관계인 명칭 매핑
  const building = (entities?.building || "").toLowerCase();
  const RESIDENTIAL = /아파트|다세대|빌라|단독주택|연립|오피스텔|주택|원룸/;
  const INDUSTRIAL   = /공장|창고|물류|제조|작업장/;
  const COMMERCIAL   = /상가|식당|음식점|카페|마트|백화점|상점|호텔|숙박/;
  const PUBLIC       = /학교|병원|관공서|도서관|체육관|복지관/;

  if (RESIDENTIAL.test(building)) {
    lines.push("건물 유형: 주거용 → 관계인은 반드시 '거주자', '세대원', '입주민'으로 지칭 ('직원', '방문객', '종업원' 사용 금지)");
  } else if (INDUSTRIAL.test(building)) {
    lines.push("건물 유형: 산업용 → 관계인은 '작업자', '관리자', '근로자'로 지칭");
  } else if (COMMERCIAL.test(building)) {
    lines.push("건물 유형: 상업용 → 관계인은 '이용객', '직원', '종업원'으로 지칭");
  } else if (PUBLIC.test(building)) {
    lines.push("건물 유형: 공공시설 → 관계인은 '이용자', '학생', '직원'으로 상황에 맞게 지칭");
  }

  // 2) 화재 발생 위치 추출 (질문 텍스트에서)
  const LOCATION_MAP = [
    [/주방|부엌|조리|가스레인지|인덕션/, "주방"],
    [/거실/, "거실"],
    [/침실|안방|방 안/, "침실"],
    [/보일러실|보일러/, "보일러실"],
    [/화장실|욕실/, "화장실"],
    [/창고|다용도실/, "창고·다용도실"],
    [/지하|지하주차장/, "지하주차장"],
    [/옥상|지붕/, "옥상"],
    [/계단|복도|공용부/, "공용부위"],
    [/차량|주차|자동차/, "주차 구역"],
  ];
  for (const [pattern, loc] of LOCATION_MAP) {
    if (pattern.test(question)) {
      lines.push(`화재 발생 위치: ${loc} → 해당 위치 특화 발화원인·조사 항목을 우선 서술`);
      break;
    }
  }

  if (lines.length === 0) return "";
  return `[조사 맥락 — 반드시 반영]\n${lines.join("\n")}\n`;
}

// =========================
// 헬퍼: userContent 생성
// =========================
function buildUserContent(question, context, rag_mode, entities) {
  const ctxNote = buildContextNote(question, entities);

  if (!context) {
    if (rag_mode === "no-data-stats" && entities?.date_ref) {
      return `[시스템 안내] 사용자가 요청한 날짜(${entities.date_ref})의 화재 통계 데이터가 데이터베이스에 없습니다. 이 사실을 먼저 명확히 알리고, 통계 조회 방법이나 대략적인 일반 정보를 안내하세요.\n\n사용자 질문: ${question}`;
    }
    if (rag_mode === "no-data-case") {
      return `[시스템 안내] 해당 조건의 화재 사례 데이터가 데이터베이스에 없습니다. 이 사실을 명확히 알리고, 전문 지식을 바탕으로 답변하세요.\n\n${ctxNote}사용자 질문: ${question}`;
    }
    if (rag_mode === "no-data-document") {
      return `[시스템 안내] 업로드된 화재조사 문서에서 관련 내용을 찾지 못했습니다. 보유한 전문 지식을 바탕으로 답변하되, 문서 기반 정보가 아님을 명시하세요.\n\n${ctxNote}사용자 질문: ${question}`;
    }
    return `${ctxNote}${question}`;
  }

  if (rag_mode === "rag-doc") {
    return `아래는 화재조사 문서에서 검색된 관련 내용입니다. 이 내용을 근거로 질문에 답변하세요.\n\n${context}\n\n${ctxNote}사용자 질문: ${question}`;
  }

  // 전체 건수 안내가 context에 있으면 user content에도 명시 (LLM이 무시하지 않도록)
  const totalMatch = context.match(/\[안내: 전체 (\d+)건 중 .+ 상위 (\d+)건을 표시합니다\]/);
  const totalInstruction = totalMatch
    ? `[중요] 아래 데이터는 전체 ${totalMatch[1]}건 중 피해 규모 기준 상위 ${totalMatch[2]}건입니다. 답변 첫 문장에 "전체 ${totalMatch[1]}건 중 피해 규모가 큰 ${totalMatch[2]}건"임을 명시하세요.\n\n`
    : "";

  return `아래는 사용자 질문과 관련된 데이터베이스 조회 결과입니다. 이 데이터가 바로 사용자가 찾는 정보입니다. 분류명(예: 창고시설, 공동주택)이 사용자 표현(예: 물류센터, 아파트)과 달라도 동일한 데이터로 취급하고 반드시 답변에 활용하세요.\n\n${totalInstruction}<db_result>\n${context}\n</db_result>\n\n사용자 질문: ${question}`;
}

// =========================
// 헬퍼: 메시지 배열 생성
// =========================
function buildMessages(systemPrompt, history, userContent) {
  const messages = [{ role: "system", content: systemPrompt }];
  if (Array.isArray(history)) {
    history.forEach((turn) => {
      if (turn.user) messages.push({ role: "user", content: turn.user });
      if (turn.assistant) {
        // 히스토리 어시스턴트 응답 길이 제한 (컨텍스트 윈도우 초과 방지)
        const trimmed = turn.assistant.length > 600
          ? turn.assistant.slice(0, 600) + "..."
          : turn.assistant;
        messages.push({ role: "assistant", content: trimmed });
      }
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
    await _searchAndBuild(question, history);

  const messages = buildMessages(systemPrompt, history, buildUserContent(q, context, rag_mode, entities));
  // 통계 모드는 일관된 수치 답변을 위해 낮은 temperature 강제
  const effectiveTemp = rag_mode === "sql-stats" ? Math.min(temperature, 0.1) : temperature;
  const txt = await callLLMStream(messages, { maxTokens: max_new_tokens, temperature: effectiveTemp, top_p });

  return { answer: txt, sources, rag_mode };
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
    await _searchAndBuild(question, history);

  const messages = buildMessages(systemPrompt, history, buildUserContent(q, context, rag_mode, entities));
  const effectiveTemp = rag_mode === "sql-stats" ? Math.min(temperature, 0.1) : temperature;
  await callLLMStreamTokens(
    messages,
    { maxTokens: max_new_tokens, temperature: effectiveTemp, top_p },
    (token) => onToken(token),
  );

  return { sources, rag_mode };
}

export default { runRag, runRagStream };
