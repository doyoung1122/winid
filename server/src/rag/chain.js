import {
  RunnableSequence,
  RunnableBranch,
  RunnableLambda,
} from "@langchain/core/runnables";

import { getEmbedding } from "../services/embedding.service.js";
import { callLLMStream, callLLMStreamTokens } from "../services/llm.service.js";
import { matchDocuments } from "../services/vector.service.js";
import { classifyIntent } from "./intent.js";
import {
  SYSTEM_RAG,
  SYSTEM_STATS,
  SYSTEM_TABLE,
  SYSTEM_SMALLTALK,
  SYSTEM_GENERAL,
  smalltalkRe,
} from "./prompts.js";
import {
  RETRIEVE_MIN,
  USE_AS_CTX_MIN,
  MIN_TOP3_AVG,
  TEXT_K,
  TABLE_K,
  IMAGE_K,
  STATS_K,
  CASE_K,
} from "../config/env.js";

// =========================
// 헬퍼: 유사도 통계
// =========================
function calcStats(matches) {
  const sims = matches.map((m) => m.similarity ?? 0);
  const maxSim = sims.length ? Math.max(...sims) : 0;
  const sorted = [...sims].sort((a, b) => b - a);
  const top3Avg = sorted.slice(0, 3).reduce((a, b) => a + b, 0) / (Math.min(sorted.length, 3) || 1);
  return { maxSim, top3Avg };
}

// =========================
// 헬퍼: 컨텍스트 문자열 생성
// =========================
function buildCtxString(matches, maxLen = 4000, tagName = "document") {
  let ctx = "";
  let length = 0;
  const sources = [];

  for (const m of matches) {
    let t = m.content?.trim();
    if (!t) continue;
    if (t.length > 1600) t = t.slice(0, 800) + "\n...\n" + t.slice(-800);
    if (length + t.length > maxLen) break;

    const meta = typeof m.metadata === "string" ? JSON.parse(m.metadata) : m.metadata || {};

    if (tagName === "stats") {
      const statType = meta.stat_type || "";
      const year = meta.year || "";
      const month = meta.month || "";
      ctx += `<stats stat_type="${statType}" year="${year}" month="${month}">\n${t}\n</stats>\n\n`;
    } else if (meta.doc_type === "case") {
      const year = meta.year || "";
      const region = meta.region || "";
      const cause = meta.cause_main || "";
      ctx += `<case year="${year}" region="${region}" cause="${cause}">\n${t}\n</case>\n\n`;
      sources.push({ filename: `사례_${year}_${region}`, similarity: m.similarity, type: "case" });
    } else {
      const filename = (meta.filepath || "").split(/[\\/]/).pop();
      const type = meta.type || "unknown";
      const page = meta.page || "N/A";
      ctx += `<document source="${filename}" page="${page}" type="${type}">\n${t}\n</document>\n\n`;
      sources.push({ filename, similarity: m.similarity, page: meta.page, type });
    }

    length += t.length;
  }

  return { ctx: ctx.trim(), sources };
}

// =========================
// 1) 질문 임베딩 + 4종 문서 검색
// =========================
const embedQuestion = RunnableLambda.from(async (input) => {
  const { question, match_count = TEXT_K, doc_sha = null } = input;

  const qVec = await getEmbedding(question, "query");

  const baseOpts = {
    threshold: RETRIEVE_MIN,
    ...(doc_sha ? { sha256: doc_sha } : {}),
  };

  // 5종 병렬 검색
  const [textMatches, tableMatches, imageMatches, statsMatches, caseMatches] = await Promise.all([
    matchDocuments(qVec, { ...baseOpts, k: match_count || TEXT_K, types: ["pdf", "text", "office", "hwpx", "hwp"] }),
    matchDocuments(qVec, { ...baseOpts, k: TABLE_K, types: ["table_row"] }),
    matchDocuments(qVec, { ...baseOpts, k: IMAGE_K, types: ["image_caption"] }),
    matchDocuments(qVec, { ...baseOpts, k: STATS_K, doc_type: "stats" }),
    matchDocuments(qVec, { ...baseOpts, k: CASE_K,  doc_type: "case" }),
  ]);

  const regularMatches = [...textMatches, ...tableMatches, ...imageMatches, ...caseMatches];

  const { maxSim: regularMaxSim, top3Avg: regularTop3Avg } = calcStats(regularMatches);
  const { maxSim: statsMaxSim } = calcStats(statsMatches);

  return { ...input, qVec, regularMatches, statsMatches, regularMaxSim, regularTop3Avg, statsMaxSim };
});

// =========================
// 2) 인텐트 분류
// =========================
const classifyIntentNode = RunnableLambda.from(async (input) => {
  const { question } = input;
  if (smalltalkRe.test(question.trim())) {
    return { ...input, intent: "smalltalk" };
  }
  const intent = await classifyIntent(question);
  return { ...input, intent };
});

// =========================
// 3) 컨텍스트 + 시스템 프롬프트 결정
// =========================
const buildContextNode = RunnableLambda.from(async (input) => {
  const { intent, regularMatches, statsMatches, regularMaxSim, regularTop3Avg, statsMaxSim } = input;

  let systemPrompt = SYSTEM_GENERAL;
  let context = "";
  let sources = [];
  let rag_mode = "general";

  if (intent === "smalltalk") {
    systemPrompt = SYSTEM_SMALLTALK;
    rag_mode = "smalltalk";

  } else if (intent === "stats" && statsMaxSim >= USE_AS_CTX_MIN) {
    // 통계 문서가 충분히 유사 → stats 모드
    const { ctx, sources: s } = buildCtxString(statsMatches, 5000, "stats");
    systemPrompt = SYSTEM_STATS;
    context = ctx;
    sources = s;
    rag_mode = "rag-stats";

  } else if (intent === "stats" && regularMaxSim >= USE_AS_CTX_MIN) {
    // 통계 질문인데 일반 문서에서 답 가능
    const { ctx, sources: s } = buildCtxString(regularMatches);
    systemPrompt = SYSTEM_RAG;
    context = ctx;
    sources = s;
    rag_mode = "rag-plain";

  } else if (intent === "stats") {
    // 통계 질문이지만 관련 데이터 없음 → stats 데이터 적재 유도
    systemPrompt = SYSTEM_GENERAL;
    rag_mode = "general-stats";

  } else if (intent === "table" && (regularMaxSim >= USE_AS_CTX_MIN || regularTop3Avg >= MIN_TOP3_AVG)) {
    const { ctx, sources: s } = buildCtxString(regularMatches);
    systemPrompt = SYSTEM_TABLE;
    context = ctx;
    sources = s;
    rag_mode = "rag-table";

  } else if (regularMaxSim >= USE_AS_CTX_MIN || regularTop3Avg >= MIN_TOP3_AVG) {
    const { ctx, sources: s } = buildCtxString(regularMatches);
    systemPrompt = SYSTEM_RAG;
    context = ctx;
    sources = s;
    rag_mode = "rag-plain";

  }
  // else: SYSTEM_GENERAL (문서 없음)

  return { ...input, systemPrompt, context, sources, rag_mode };
});

// =========================
// 4) LLM 호출 (메인 답변)
// =========================
const callLLMNode = RunnableLambda.from(async (input) => {
  const { systemPrompt, context, question, history, max_new_tokens, temperature, top_p } = input;

  const messages = [{ role: "system", content: systemPrompt }];

  if (Array.isArray(history)) {
    history.forEach((turn) => {
      if (turn.user) messages.push({ role: "user", content: turn.user });
      if (turn.assistant) messages.push({ role: "assistant", content: turn.assistant });
    });
  }

  const userContent = context
    ? `아래에 참고 데이터가 제공됩니다. 내용을 분석하여 사용자 질문에 답변하세요.\n\n<context>\n${context}\n</context>\n\n사용자 질문: ${question}`
    : question;

  messages.push({ role: "user", content: userContent });

  const txt = await callLLMStream(messages, { maxTokens: max_new_tokens, temperature, top_p });

  return { ...input, answer: txt };
});

// =========================
// 5) 최종 출력 포맷
// =========================
const formatOutput = RunnableLambda.from((input) => ({
  answer: input.answer,
  sources: input.sources || [],
  rag_mode: input.rag_mode,
}));

// =========================
// 6) 최종 실행 함수 (non-streaming)
// =========================
export async function runRag({
  question,
  history = [],
  match_count = TEXT_K,
  max_new_tokens = 800,
  temperature = 0.3,
  top_p = 0.9,
  doc_sha = null,
}) {
  const base = { question, history, match_count, max_new_tokens, temperature, top_p, doc_sha };
  const chain = RunnableSequence.from([
    embedQuestion,
    classifyIntentNode,
    buildContextNode,
    callLLMNode,
    formatOutput,
  ]);
  return await chain.invoke(base);
}

// =========================
// 7) SSE 스트리밍 실행 함수
// =========================
export async function runRagStream({
  question,
  history = [],
  match_count = TEXT_K,
  max_new_tokens = 800,
  temperature = 0.3,
  top_p = 0.9,
  doc_sha = null,
  onToken,
}) {
  // 1. 임베딩 + 4종 병렬 검색
  const qVec = await getEmbedding(question, "query");
  const baseOpts = { threshold: RETRIEVE_MIN, ...(doc_sha ? { sha256: doc_sha } : {}) };

  const [textMatches, tableMatches, imageMatches, statsMatches, caseMatches] = await Promise.all([
    matchDocuments(qVec, { ...baseOpts, k: match_count || TEXT_K, types: ["pdf", "text", "office", "hwpx", "hwp"] }),
    matchDocuments(qVec, { ...baseOpts, k: TABLE_K, types: ["table_row"] }),
    matchDocuments(qVec, { ...baseOpts, k: IMAGE_K, types: ["image_caption"] }),
    matchDocuments(qVec, { ...baseOpts, k: STATS_K, doc_type: "stats" }),
    matchDocuments(qVec, { ...baseOpts, k: CASE_K,  doc_type: "case" }),
  ]);

  const regularMatches = [...textMatches, ...tableMatches, ...imageMatches, ...caseMatches];
  const { maxSim: regularMaxSim, top3Avg: regularTop3Avg } = calcStats(regularMatches);
  const { maxSim: statsMaxSim } = calcStats(statsMatches);

  // 2. 인텐트 분류
  let intent;
  if (smalltalkRe.test(question.trim())) {
    intent = "smalltalk";
  } else {
    intent = await classifyIntent(question);
  }

  // 3. 컨텍스트 + 프롬프트 결정
  let systemPrompt = SYSTEM_GENERAL;
  let ctx = "";
  let sources = [];
  let rag_mode = "general";

  if (intent === "smalltalk") {
    systemPrompt = SYSTEM_SMALLTALK;
    rag_mode = "smalltalk";

  } else if (intent === "stats" && statsMaxSim >= USE_AS_CTX_MIN) {
    const { ctx: c, sources: s } = buildCtxString(statsMatches, 5000, "stats");
    systemPrompt = SYSTEM_STATS;
    ctx = c;
    sources = s;
    rag_mode = "rag-stats";

  } else if (intent === "stats" && regularMaxSim >= USE_AS_CTX_MIN) {
    const { ctx: c, sources: s } = buildCtxString(regularMatches);
    systemPrompt = SYSTEM_RAG;
    ctx = c;
    sources = s;
    rag_mode = "rag-plain";

  } else if (intent === "stats") {
    systemPrompt = SYSTEM_GENERAL;
    rag_mode = "general-stats";

  } else if (intent === "table" && (regularMaxSim >= USE_AS_CTX_MIN || regularTop3Avg >= MIN_TOP3_AVG)) {
    const { ctx: c, sources: s } = buildCtxString(regularMatches);
    systemPrompt = SYSTEM_TABLE;
    ctx = c;
    sources = s;
    rag_mode = "rag-table";

  } else if (regularMaxSim >= USE_AS_CTX_MIN || regularTop3Avg >= MIN_TOP3_AVG) {
    const { ctx: c, sources: s } = buildCtxString(regularMatches);
    systemPrompt = SYSTEM_RAG;
    ctx = c;
    sources = s;
    rag_mode = "rag-plain";
  }

  // 4. 메시지 구성
  const messages = [{ role: "system", content: systemPrompt }];
  if (Array.isArray(history)) {
    history.forEach((turn) => {
      if (turn.user) messages.push({ role: "user", content: turn.user });
      if (turn.assistant) messages.push({ role: "assistant", content: turn.assistant });
    });
  }

  const userContent = ctx
    ? `아래에 참고 데이터가 제공됩니다. 내용을 분석하여 사용자 질문에 답변하세요.\n\n<context>\n${ctx}\n</context>\n\n사용자 질문: ${question}`
    : question;
  messages.push({ role: "user", content: userContent });

  // 5. LLM 스트리밍
  await callLLMStreamTokens(messages, { maxTokens: max_new_tokens, temperature, top_p }, onToken);

  return { sources, rag_mode };
}

export default { runRag, runRagStream };
