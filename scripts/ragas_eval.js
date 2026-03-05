/**
 * RAGAS-style 품질 평가 스크립트 (LLM-as-Judge)
 *
 * 측정 지표:
 *   - Faithfulness     : 답변이 컨텍스트에 근거하는가? (환각 감지)
 *   - Answer Relevance : 답변이 질문에 적절히 응답했는가?
 *   - Context Precision: 검색된 컨텍스트가 질문에 관련 있는가?
 *
 * 실행:
 *   node scripts/ragas_eval.js                        # 전체 평가
 *   node scripts/ragas_eval.js --tag=after-bm25       # 태그 붙여 저장
 *   node scripts/ragas_eval.js --type=rag-doc         # 특정 타입만
 *   node scripts/ragas_eval.js --id=d01,d02           # 특정 항목만
 *   node scripts/ragas_eval.js --compare=before,after # 두 결과 비교
 */

import "dotenv/config";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── 설정 ──────────────────────────────────────────
const SERVER_URL  = process.env.EVAL_SERVER_URL || "http://localhost:8000";
// 평가 스크립트는 호스트에서 실행 → localhost 직접 접근
// Docker 내부용 host.docker.internal은 호스트에서 사용 불가
const LLM_URL     = (process.env.EVAL_LLM_URL || process.env.LLM_URL || "http://localhost:8002")
  .replace("host.docker.internal", "localhost")
  .replace(/\/$/, "");
const LLM_MODEL   = process.env.LLM_MODEL || "qwen3";
const RESULTS_DIR = join(__dirname, "../eval_results");
const DATASET_PATH = join(__dirname, "eval_dataset.json");

// CLI 인수 파싱
const args = Object.fromEntries(
  process.argv.slice(2)
    .filter(a => a.startsWith("--"))
    .map(a => { const [k, v] = a.slice(2).split("="); return [k, v ?? true]; })
);

if (!existsSync(RESULTS_DIR)) mkdirSync(RESULTS_DIR, { recursive: true });

// ── LLM Judge 호출 ────────────────────────────────
async function llmJudge(prompt) {
  const r = await fetch(`${LLM_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 10,
      temperature: 0.0,
      chat_template_kwargs: { enable_thinking: false },
    }),
  });
  if (!r.ok) throw new Error(`LLM judge failed: ${r.status}`);
  const j = await r.json();
  const text = j.choices[0]?.message?.content?.trim() || "";
  // 숫자 추출 (0.0 ~ 1.0)
  const m = text.match(/([01](?:\.\d+)?)/);
  return m ? Math.min(1, Math.max(0, parseFloat(m[1]))) : null;
}

// ── 지표별 프롬프트 ────────────────────────────────
function promptFaithfulness(question, context, answer) {
  return `당신은 화재 감식 AI 답변의 품질을 평가하는 전문 평가자입니다.

[질문]
${question}

[검색된 컨텍스트]
${context.slice(0, 2000)}

[AI 답변]
${answer.slice(0, 1000)}

평가: AI 답변의 주장이 컨텍스트에 의해 뒷받침되는가?
- 1.0: 모든 주장이 컨텍스트에 근거함
- 0.5: 일부 주장이 컨텍스트 외부 정보 포함
- 0.0: 컨텍스트와 무관하거나 허위 정보

0.0~1.0 사이 숫자만 출력. 설명 없이.`;
}

function promptAnswerRelevance(question, answer) {
  return `당신은 화재 감식 AI 답변의 품질을 평가하는 전문 평가자입니다.

[질문]
${question}

[AI 답변]
${answer.slice(0, 1000)}

평가: 답변이 질문을 얼마나 잘 다루는가?
- 1.0: 질문의 핵심을 정확하고 완전하게 답변
- 0.5: 부분적으로 답변하거나 일부 누락
- 0.0: 질문과 무관하거나 답변 거부

0.0~1.0 사이 숫자만 출력. 설명 없이.`;
}

function promptContextPrecision(question, context) {
  return `당신은 화재 감식 AI 시스템의 검색 품질을 평가하는 전문 평가자입니다.

[질문]
${question}

[검색된 컨텍스트]
${context.slice(0, 2000)}

평가: 검색된 컨텍스트가 질문 답변에 얼마나 유용한가?
- 1.0: 대부분의 컨텍스트가 질문 답변에 직접적으로 유용
- 0.5: 절반 정도 유용, 일부 무관한 내용 포함
- 0.0: 거의 모든 컨텍스트가 질문과 무관

0.0~1.0 사이 숫자만 출력. 설명 없이.`;
}

// ── /query 호출 ────────────────────────────────────
async function queryServer(question) {
  const r = await fetch(`${SERVER_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      include_context: true,
      max_new_tokens: 800,
      temperature: 0.2,
    }),
  });
  if (!r.ok) throw new Error(`/query failed: ${r.status}`);
  return r.json();
}

// ── 단일 항목 평가 ─────────────────────────────────
async function evaluateOne(item) {
  const start = Date.now();
  process.stdout.write(`  [${item.id}] "${item.question.slice(0, 30)}..." `);

  let queryResult;
  try {
    queryResult = await queryServer(item.question);
  } catch (e) {
    console.log(`❌ query 실패: ${e.message}`);
    return { ...item, error: e.message, scores: null };
  }

  const { answer = "", context = "", rag_mode } = queryResult;
  const hasContext = context && context.trim().length > 20;

  // 지표 채점 (병렬)
  const [faithfulness, answerRelevance, contextPrecision] = await Promise.all([
    hasContext ? llmJudge(promptFaithfulness(item.question, context, answer)).catch(() => null) : null,
    llmJudge(promptAnswerRelevance(item.question, answer)).catch(() => null),
    hasContext ? llmJudge(promptContextPrecision(item.question, context)).catch(() => null) : null,
  ]);

  const modeMatch = rag_mode === item.expected_mode ||
    // sql-case와 rag-doc은 데이터 유무에 따라 달라질 수 있어 허용
    (item.expected_mode === "sql-case" && ["sql-case", "rag-doc", "no-data-case"].includes(rag_mode));

  const scores = { faithfulness, answerRelevance, contextPrecision };
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);

  const scoreStr = [
    faithfulness   != null ? `F=${faithfulness.toFixed(2)}`   : "F=n/a",
    answerRelevance!= null ? `R=${answerRelevance.toFixed(2)}` : "R=n/a",
    contextPrecision!= null? `C=${contextPrecision.toFixed(2)}`: "C=n/a",
  ].join(" ");

  console.log(`${modeMatch ? "✅" : "⚠️ "} ${rag_mode} | ${scoreStr} | ${elapsed}s`);

  return {
    id: item.id,
    type: item.type,
    question: item.question,
    expected_mode: item.expected_mode,
    actual_mode: rag_mode,
    mode_match: modeMatch,
    answer_preview: answer.slice(0, 100),
    scores,
    elapsed_ms: Date.now() - start,
  };
}

// ── 집계 ───────────────────────────────────────────
function aggregate(results) {
  const valid = results.filter(r => r.scores);
  const avg = (key) => {
    const vals = valid.map(r => r.scores[key]).filter(v => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };
  const byType = {};
  for (const type of ["stats", "case", "rag-doc"]) {
    const sub = valid.filter(r => r.type === type);
    byType[type] = {
      count: sub.length,
      mode_match_rate: sub.filter(r => r.mode_match).length / (sub.length || 1),
      faithfulness:    (() => { const v = sub.map(r => r.scores.faithfulness).filter(x => x != null); return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null; })(),
      answerRelevance: (() => { const v = sub.map(r => r.scores.answerRelevance).filter(x => x != null); return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null; })(),
      contextPrecision:(() => { const v = sub.map(r => r.scores.contextPrecision).filter(x => x != null); return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null; })(),
    };
  }
  return {
    total: results.length,
    errors: results.filter(r => r.error).length,
    mode_match_rate: valid.filter(r => r.mode_match).length / (valid.length || 1),
    faithfulness:    avg("faithfulness"),
    answerRelevance: avg("answerRelevance"),
    contextPrecision:avg("contextPrecision"),
    overall: (() => {
      const vals = [avg("faithfulness"), avg("answerRelevance"), avg("contextPrecision")].filter(v => v != null);
      return vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
    })(),
    byType,
  };
}

// ── 결과 출력 ──────────────────────────────────────
function printSummary(summary, tag) {
  const fmt = (v) => v != null ? v.toFixed(3) : " n/a ";
  const pct = (v) => v != null ? `${(v*100).toFixed(1)}%` : " n/a ";

  console.log("\n" + "═".repeat(60));
  console.log(`  RAGAS 평가 결과${tag ? ` [${tag}]` : ""}`);
  console.log("═".repeat(60));
  console.log(`  전체 ${summary.total}건 | 오류 ${summary.errors}건 | 모드 정확도 ${pct(summary.mode_match_rate)}`);
  console.log("─".repeat(60));
  console.log(`  ${"지표".padEnd(20)} ${"전체".padStart(6)} ${"stats".padStart(6)} ${"case".padStart(6)} ${"rag-doc".padStart(7)}`);
  console.log("─".repeat(60));

  const metrics = [
    ["Faithfulness",     "faithfulness"],
    ["Answer Relevance", "answerRelevance"],
    ["Context Precision","contextPrecision"],
  ];
  for (const [label, key] of metrics) {
    const overall = fmt(summary[key]);
    const s = fmt(summary.byType.stats?.[key]);
    const c = fmt(summary.byType.case?.[key]);
    const d = fmt(summary.byType["rag-doc"]?.[key]);
    console.log(`  ${label.padEnd(20)} ${overall.padStart(6)} ${s.padStart(6)} ${c.padStart(6)} ${d.padStart(7)}`);
  }
  console.log("─".repeat(60));
  console.log(`  ${"Overall".padEnd(20)} ${fmt(summary.overall).padStart(6)}`);
  console.log("═".repeat(60) + "\n");
}

// ── 비교 모드 ──────────────────────────────────────
function compareResults(tags) {
  const loaded = tags.map(tag => {
    const path = join(RESULTS_DIR, `${tag}.json`);
    if (!existsSync(path)) { console.error(`파일 없음: ${path}`); process.exit(1); }
    return { tag, data: JSON.parse(readFileSync(path, "utf8")) };
  });

  const metrics = ["faithfulness", "answerRelevance", "contextPrecision", "overall"];
  console.log("\n" + "═".repeat(70));
  console.log("  RAGAS 비교");
  console.log("═".repeat(70));
  console.log(`  ${"지표".padEnd(22)} ${loaded.map(l => l.tag.padStart(12)).join(" ")}`);
  console.log("─".repeat(70));
  for (const m of metrics) {
    const vals = loaded.map(l => {
      const v = l.data.summary[m];
      return (v != null ? v.toFixed(3) : " n/a ").padStart(12);
    });
    console.log(`  ${m.padEnd(22)} ${vals.join(" ")}`);
  }
  console.log("═".repeat(70) + "\n");
}

// ── 메인 ───────────────────────────────────────────
async function main() {
  // 비교 모드
  if (args.compare) {
    compareResults(args.compare.split(",").map(s => s.trim()));
    return;
  }

  // 데이터셋 로드
  const dataset = JSON.parse(readFileSync(DATASET_PATH, "utf8"));

  // 필터링
  let items = dataset;
  if (args.type)  items = items.filter(i => i.type === args.type);
  if (args.id)    items = items.filter(i => args.id.split(",").includes(i.id));

  const tag = args.tag || new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  console.log(`\n🔍 RAGAS 평가 시작 | ${items.length}건 | tag=${tag}`);
  console.log(`   서버: ${SERVER_URL} | LLM: ${LLM_URL}`);
  console.log(`   지표: F=Faithfulness R=AnswerRelevance C=ContextPrecision\n`);

  // 순차 실행 (LLM 과부하 방지)
  const results = [];
  for (const item of items) {
    const r = await evaluateOne(item);
    results.push(r);
  }

  const summary = aggregate(results);
  printSummary(summary, tag);

  // 결과 저장
  const output = { tag, timestamp: new Date().toISOString(), summary, results };
  const outPath = join(RESULTS_DIR, `${tag}.json`);
  writeFileSync(outPath, JSON.stringify(output, null, 2), "utf8");
  console.log(`💾 결과 저장: ${outPath}\n`);
}

main().catch(e => { console.error("평가 실패:", e); process.exit(1); });
