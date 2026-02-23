import { callLLM, withTimeout } from "../services/llm.service.js";
import { tableLikeRe, statsRe } from "./prompts.js";
import { LLM_MODEL } from "../config/env.js";

/**
 * Classify the intent of a question
 * @param {string} question - User question
 * @returns {Promise<"plain"|"table"|"stats">} Intent type
 */
export async function classifyIntent(question) {
  // 1. 명시적으로 "표"라고 말했으면 바로 표 모드 (속도 최적화)
  if (tableLikeRe.test(question)) return "table";

  // 2. 통계/추이 관련 키워드가 있으면 stats 모드
  if (statsRe.test(question)) return "stats";

  // 3. LLM으로 판단 (짧은 질문, 키워드 없는 경우)
  try {
    const answer = await withTimeout(
      callLLM(
        [
          {
            role: "system",
            content: `사용자의 질문 유형을 분류하세요.

유형 정의:
- stats: 화재 통계, 건수 추이, 월별/연도별 변화, 발화요인별 집계, 피해액 동향 등 수치 데이터 분석 질문
- table: 표 형식으로 출력을 원하는 질문
- plain: 개념 설명, 감식 방법론, 사례 분석, 일반 지식 등

반드시 세 단어 중 하나만 출력하세요: stats, table, plain`,
          },
          { role: "user", content: question },
        ],
        { temperature: 0.0, maxTokens: 10 }
      ),
      5000,
      "intent"
    );

    const intent = answer.trim().toLowerCase();
    if (intent.includes("stats")) return "stats";
    if (intent.includes("table")) return "table";
    return "plain";
  } catch (e) {
    console.error("intent check failed, defaulting to plain:", e.message);
    return "plain";
  }
}

export default { classifyIntent };
