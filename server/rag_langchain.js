require("dotenv").config();
const fetch = require("node-fetch");
const {
  RunnableSequence,
  RunnableBranch,
  RunnableLambda,
} = require("@langchain/core/runnables");

const { matchDocuments } = require("../db/repo.js");

// =========================
// 환경변수 / URL 설정
// =========================
const EMB_URL = (process.env.EMB_URL || "http://127.0.0.1:8001").replace(/\/$/, "");
const LLM_URL = (process.env.LLM_URL || "http://127.0.0.1:8002").replace(/\/$/, "");

// vLLM 모델명
const LLM_MODEL_NAME = "qwen-coder"; 
const EMB_MODEL_NAME = "BAAI/bge-m3";

// =========================
// RAG 검색 품질 기준값
// =========================
const RETRIEVE_MIN   = 0.35;
const USE_AS_CTX_MIN = 0.60;
const MIN_TOP3_AVG   = 0.55;

const TEXT_K  = 5;
const TABLE_K = 10;
const IMAGE_K = 4;

// =========================
// 정규식 (빠른 분류용)
// =========================
const smalltalkRe = /^(?:\s*)(hi|hello|안녕|하이|헬로|감사|땡큐|잘가|bye|바이|누구야|너 누구|자기소개|도움)(?:[\s!?.…]*)$/i;
const tableLikeRe = /(표로|표 형태|table|테이블|표를|표 형식|표 형식으로|표로 정리|표로 보여줘|표에|표에서|표 안에|표에 정리된|표에 나온)/i;

// =========================
// 유틸리티 함수
// =========================
function withTimeout(promise, ms = 30000, tag = "req") {
  return Promise.race([
    promise,
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error(`${tag} timeout ${ms}ms`)), ms)
    ),
  ]);
}

async function readSSEToText(readable) {
  return await new Promise((resolve, reject) => {
    let acc = "";
    readable.on("data", (buf) => {
      const chunk = buf.toString();
      const lines = chunk.split(/\r?\n/);
      for (const line of lines) {
        if (!line) continue;
        if (line.startsWith("data: ")) {
          const payload = line.slice(6);
          if (payload === "[DONE]") continue;
          try {
            const json = JSON.parse(payload);
            const content = json.choices?.[0]?.delta?.content || "";
            acc += content;
          } catch (e) { }
        }
      }
    });
    readable.on("end", () => resolve(acc));
    readable.on("error", (err) => reject(err));
  });
}

// =========================================================
// 1) 의도 파악
// =========================================================
async function classifyIntent(question) {
  // 1. 명시적으로 "표"라고 말했으면 바로 표 모드 (속도 최적화)
  if (tableLikeRe.test(question)) return "table";

  // 2. 말 안 했어도 상황 판단 (LLM에게 물어봄)
  try {
    const res = await withTimeout(
      fetch(`${LLM_URL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: LLM_MODEL_NAME,
          messages: [
            {
              role: "system",
              content: `Determine the best output format for the user's question.
Choose between 'plain' (text explanation) or 'table' (structured data).

# Rules for 'table'
Select 'table' if the user asks for:
1. **Comparisons** (e.g., "Difference between A and B", "Compare X and Y")
2. **Lists of specs/stats** (e.g., "List the melting points", "Show the capacities")
3. **Timeline/Schedule** (e.g., "History of...", "Event logs")

# Output
Output ONLY one word: 'plain' or 'table'.`
            },
            { role: "user", content: question }
          ],
          max_tokens: 10,
          temperature: 0.0,
          stream: false
        }),
      }),
      5000,
      "intent"
    );

    if (!res.ok) return "plain";
    const data = await res.json();
    const text = data.choices?.[0]?.message?.content || "";
    const intent = text.trim().toLowerCase();

    if (intent.includes("table")) return "table";
    return "plain";

  } catch (e) {
    console.error("Smart intent check failed, defaulting to plain:", e.message);
    return "plain";
  }
}

// =========================
// 2) 질문 임베딩 생성
// =========================
const embedQuestion = RunnableLambda.from(async (input) => {
  const { question, match_count = TEXT_K, doc_sha = null } = input;

  const embRes = await withTimeout(
    fetch(`${EMB_URL}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: EMB_MODEL_NAME,
        input: question
      }),
    }),
    20000,
    "embed(q)"
  );

  if (!embRes.ok) throw new Error(`Embedding fail: ${embRes.status}`);
  const embJson = await embRes.json();
  const qVec = embJson.data[0].embedding;

  const baseOpts = {
    threshold: RETRIEVE_MIN,
    ...(doc_sha ? { sha256: doc_sha } : {}),
  };

  const textMatches = await matchDocuments(qVec, { ...baseOpts, k: match_count || TEXT_K, types: ["pdf", "text", "office", "hwpx", "hwp"] });
  const tableMatches = await matchDocuments(qVec, { ...baseOpts, k: TABLE_K, types: ["table_row"] });
  const imageMatches = await matchDocuments(qVec, { ...baseOpts, k: IMAGE_K, types: ["image_caption"] });

  const matches = [...textMatches, ...tableMatches, ...imageMatches];
  const sims = matches.map((m) => m.similarity ?? 0);
  const maxSim = sims.length ? Math.max(...sims) : 0;

  return { ...input, qVec, matches, maxSim };
});

// =========================
// 3) 검색 결과 통계
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
// 4) 컨텍스트 구성
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
// 5) LLM 호출 (메인 답변)
// =========================
const callLLM = RunnableLambda.from(async (input) => {
  const { systemPrompt, context, question, history, max_new_tokens, temperature, top_p } = input;

  const messages = [
    { role: "system", content: systemPrompt }
  ];

  if (Array.isArray(history)) {
    history.forEach(turn => {
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

  const llm = await fetch(`${LLM_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      model: LLM_MODEL_NAME,
      messages: messages,
      max_tokens: max_new_tokens,
      temperature,
      top_p,
      stream: true
    }),
  });

  if (!llm.ok) {
     const err = await llm.text();
     throw new Error(`LLM call failed: ${llm.status} ${err}`);
  }

  const txt = await readSSEToText(llm.body);
  return { ...input, answer: txt.trim() };
});

// =================================================================
// 6) 시스템 프롬프트 (영어 지시 + 한국어 출력)
// =================================================================

// (1) 일반 모드 (Text)
const SYSTEM_PLAIN = `
You are an intelligent expert specialized in Fire Investigation (화재 감식).

# Goal
Answer the user's question based strictly on the provided <context>.

# Rules
1. **Language:** ALWAYS answer in **Korean (한국어)**.
2. **Grounding:** Answer ONLY based on the provided <context>.
3. **Safety:** If the answer is not found, state: "제공된 문서에서 관련 내용을 찾을 수 없습니다."
4. **Tone:** Professional and objective.
`.trim();

// (2) 표 모드 (Table)
const SYSTEM_TABLE = `
You are an AI assistant specialized in structuring data into Tables.

# Rules
1. **Language:** ALWAYS answer in **Korean (한국어)**.
2. **Task:** Construct a **Markdown Table** based strictly on the provided <context>.
3. **Format:** Ensure the table has clear headers and rows.
4. **Missing Data:** Use "-" for missing values.
5. **Summary:** Add a short Korean summary below the table.
`.trim();

// (3) 잡담 (Smalltalk)
const SYSTEM_SMALLTALK = `
You are a polite AI assistant.

# Rules
1. **Language:** ALWAYS answer in **Korean (한국어)**.
2. **Constraint:** Do NOT mention documents. Respond naturally to the greeting.
`.trim();

// (4) 일반 지식 (No Context)
const SYSTEM_GENERAL = `
You are an honest AI assistant.
Currently, **NO reference documents are available** to answer this question.

# STRICT RULES (Follow these or stop generation)
1. **Scope:** You can ONLY answer questions about **general concepts, definitions, or scientific principles** (e.g., "What is Flashover?", "Explain Ohm's Law").
2. **Prohibition:** NEVER answer questions about:
   - Specific events, dates, or locations.
   - Specific people or organizations.
   - Specific laws or regulations (unless you are 100% sure of the exact text).
   - Numerical data or statistics.
3. **Refusal Strategy:** If the user asks for specific facts without documents, explicitly reply in Korean:
   - "현재 참고할 문서가 없어 정확한 답변을 드릴 수 없습니다."
   - or "해당 내용은 일반적인 지식으로 답변하기 어렵습니다. 문서를 업로드해주세요."
4. **Tone:** Humble and cautious. Do not pretend to know details.
5. **Language:** ALWAYS answer in **Korean (한국어)**.
`.trim();


// =========================
// 7) 분기 로직
// =========================
const autoBranch = RunnableBranch.from([
  // 1. 잡담
  [
    (input) => smalltalkRe.test(input.question.trim()),
    RunnableSequence.from([
      RunnableLambda.from(async (input) => ({ ...input, systemPrompt: SYSTEM_SMALLTALK, context: "" })),
      callLLM,
      (input) => ({ answer: input.answer, sources: [], rag_mode: "smalltalk" }),
    ]),
  ],
  // 2. 문서 기반 RAG
  [
    (input) => { const { maxSim, top3Avg } = input; return maxSim >= USE_AS_CTX_MIN || top3Avg >= MIN_TOP3_AVG; },
    RunnableSequence.from([
      RunnableLambda.from(async (input) => {
        const intent = await classifyIntent(input.question);
        
        let systemPrompt = SYSTEM_PLAIN; 
        if (intent === "table") systemPrompt = SYSTEM_TABLE;
        
        return { ...input, intent, systemPrompt };
      }),
      callLLM,
      (input) => ({ answer: input.answer, sources: input.sources, rag_mode: `rag-${input.intent}` }),
    ]),
  ],
  // 3. 문서 없음
  RunnableSequence.from([
    RunnableLambda.from(async (input) => ({ ...input, systemPrompt: SYSTEM_GENERAL, context: "" })),
    callLLM,
    (input) => ({ answer: input.answer, sources: [], rag_mode: "general" }),
  ]),
]);

// =========================
// 8) 최종 실행 함수
// =========================
async function runRag({ question, history = [], match_count = TEXT_K, max_new_tokens = 600, temperature = 0.2, top_p = 0.9, doc_sha = null }) {
  const base = { question, history, match_count, max_new_tokens, temperature, top_p, doc_sha };
  const chain = RunnableSequence.from([ embedQuestion, retrieveDocs, buildContext, autoBranch ]);
  return await chain.invoke(base);
}

module.exports = { runRag };