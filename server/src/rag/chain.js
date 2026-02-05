import {
  RunnableSequence,
  RunnableBranch,
  RunnableLambda,
} from "@langchain/core/runnables";

import { getEmbedding } from "../services/embedding.service.js";
import { callLLMStream, withTimeout } from "../services/llm.service.js";
import { matchDocuments } from "../../../db/repo.js";
import { classifyIntent } from "./intent.js";
import {
  SYSTEM_PLAIN,
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
  EMB_MODEL,
} from "../config/env.js";

// =========================
// 1) 질문 임베딩 생성
// =========================
const embedQuestion = RunnableLambda.from(async (input) => {
  const { question, match_count = TEXT_K, doc_sha = null } = input;

  const qVec = await getEmbedding(question, "query");

  const baseOpts = {
    threshold: RETRIEVE_MIN,
    ...(doc_sha ? { sha256: doc_sha } : {}),
  };

  const textMatches = await matchDocuments(qVec, {
    ...baseOpts,
    k: match_count || TEXT_K,
    types: ["pdf", "text", "office", "hwpx", "hwp"],
  });
  const tableMatches = await matchDocuments(qVec, {
    ...baseOpts,
    k: TABLE_K,
    types: ["table_row"],
  });
  const imageMatches = await matchDocuments(qVec, {
    ...baseOpts,
    k: IMAGE_K,
    types: ["image_caption"],
  });

  const matches = [...textMatches, ...tableMatches, ...imageMatches];
  const sims = matches.map((m) => m.similarity ?? 0);
  const maxSim = sims.length ? Math.max(...sims) : 0;

  return { ...input, qVec, matches, maxSim };
});

// =========================
// 2) 검색 결과 통계
// =========================
const retrieveDocs = RunnableLambda.from(async (input) => {
  const { matches } = input;
  const sims = matches.map((m) => m.similarity ?? 0);
  const sorted = [...sims].sort((a, b) => b - a);
  const top3 = sorted.slice(0, 3);
  const top3Avg = top3.length > 0 ? top3.reduce((a, b) => a + b, 0) / top3.length : 0;
  return { ...input, top3Avg };
});

// =========================
// 3) 컨텍스트 구성
// =========================
const buildContext = RunnableLambda.from(async (input) => {
  const { matches } = input;
  const MAX_CTX = 4000;
  let ctx = "";
  let length = 0;
  const sources = [];

  for (const m of matches) {
    let t = m.content?.trim();
    if (!t) continue;
    if (t.length > 1600) t = t.slice(0, 800) + "\n...\n" + t.slice(-800);
    if (length + t.length > MAX_CTX) break;

    const meta = typeof m.metadata === "string" ? JSON.parse(m.metadata) : m.metadata || {};
    const filename = (meta.filepath || "").split(/[\\/]/).pop();
    const type = meta.type || "unknown";
    const page = meta.page || "N/A";

    ctx += `<document source="${filename}" page="${page}" type="${type}">
${t}
</document>\n\n`;

    length += t.length;
    sources.push({ filename, similarity: m.similarity, page: meta.page, type });
  }
  return { ...input, context: ctx.trim(), sources };
});

// =========================
// 4) LLM 호출 (메인 답변)
// =========================
const callLLMNode = RunnableLambda.from(async (input) => {
  const { systemPrompt, context, question, history, max_new_tokens, temperature, top_p } =
    input;

  const messages = [{ role: "system", content: systemPrompt }];

  if (Array.isArray(history)) {
    history.forEach((turn) => {
      if (turn.user) messages.push({ role: "user", content: turn.user });
      if (turn.assistant) messages.push({ role: "assistant", content: turn.assistant });
    });
  }

  const userContent = context
    ? `Reference Documents are provided below in XML format.
Read them carefully and answer the user's question.

<context>
${context}
</context>

User Question: ${question}`
    : question;

  messages.push({ role: "user", content: userContent });

  const txt = await callLLMStream(messages, {
    maxTokens: max_new_tokens,
    temperature,
    top_p,
  });

  return { ...input, answer: txt };
});

// =========================
// 5) 분기 로직
// =========================
const autoBranch = RunnableBranch.from([
  // 1. 잡담
  [
    (input) => smalltalkRe.test(input.question.trim()),
    RunnableSequence.from([
      RunnableLambda.from(async (input) => ({
        ...input,
        systemPrompt: SYSTEM_SMALLTALK,
        context: "",
      })),
      callLLMNode,
      (input) => ({ answer: input.answer, sources: [], rag_mode: "smalltalk" }),
    ]),
  ],
  // 2. 문서 기반 RAG
  [
    (input) => {
      const { maxSim, top3Avg } = input;
      return maxSim >= USE_AS_CTX_MIN || top3Avg >= MIN_TOP3_AVG;
    },
    RunnableSequence.from([
      RunnableLambda.from(async (input) => {
        const intent = await classifyIntent(input.question);

        let systemPrompt = SYSTEM_PLAIN;
        if (intent === "table") systemPrompt = SYSTEM_TABLE;

        return { ...input, intent, systemPrompt };
      }),
      callLLMNode,
      (input) => ({
        answer: input.answer,
        sources: input.sources,
        rag_mode: `rag-${input.intent}`,
      }),
    ]),
  ],
  // 3. 문서 없음
  RunnableSequence.from([
    RunnableLambda.from(async (input) => ({
      ...input,
      systemPrompt: SYSTEM_GENERAL,
      context: "",
    })),
    callLLMNode,
    (input) => ({ answer: input.answer, sources: [], rag_mode: "general" }),
  ]),
]);

// =========================
// 6) 최종 실행 함수
// =========================
export async function runRag({
  question,
  history = [],
  match_count = TEXT_K,
  max_new_tokens = 600,
  temperature = 0.2,
  top_p = 0.9,
  doc_sha = null,
}) {
  const base = { question, history, match_count, max_new_tokens, temperature, top_p, doc_sha };
  const chain = RunnableSequence.from([embedQuestion, retrieveDocs, buildContext, autoBranch]);
  return await chain.invoke(base);
}

export default { runRag };
