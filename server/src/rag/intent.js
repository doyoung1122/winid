import { callLLM, withTimeout } from "../services/llm.service.js";
import { tableLikeRe } from "./prompts.js";
import { LLM_MODEL } from "../config/env.js";

/**
 * Classify the intent of a question (plain vs table)
 * @param {string} question - User question
 * @returns {Promise<"plain"|"table">} Intent type
 */
export async function classifyIntent(question) {
  // 1. 명시적으로 "표"라고 말했으면 바로 표 모드 (속도 최적화)
  if (tableLikeRe.test(question)) return "table";

  // 2. 말 안 했어도 상황 판단 (LLM에게 물어봄)
  try {
    const answer = await withTimeout(
      callLLM(
        [
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
Output ONLY one word: 'plain' or 'table'.`,
          },
          { role: "user", content: question },
        ],
        { temperature: 0.0, maxTokens: 10 }
      ),
      5000,
      "intent"
    );

    const intent = answer.trim().toLowerCase();
    if (intent.includes("table")) return "table";
    return "plain";
  } catch (e) {
    console.error("Smart intent check failed, defaulting to plain:", e.message);
    return "plain";
  }
}

export default { classifyIntent };
