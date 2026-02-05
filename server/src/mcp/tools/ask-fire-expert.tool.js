import { callLLM } from "../../services/llm.service.js";
import { SYSTEM_FIRE_EXPERT } from "../../rag/prompts.js";

export const definition = {
  name: "ask_fire_expert",
  description:
    "화재 감식 전문가에게 질문합니다. 화재 원인, 통계, 조사 방법 등에 대한 질문에 답변합니다.",
  inputSchema: {
    type: "object",
    properties: {
      question: {
        type: "string",
        description: "화재 관련 질문 (한국어)",
      },
      context: {
        type: "string",
        description: "추가 컨텍스트 또는 참고 자료 (선택)",
      },
    },
    required: ["question"],
  },
};

export async function handler({ question, context }) {
  const messages = [{ role: "system", content: SYSTEM_FIRE_EXPERT }];

  if (context) {
    messages.push({
      role: "user",
      content: `참고 자료:\n${context}\n\n질문: ${question}`,
    });
  } else {
    messages.push({ role: "user", content: question });
  }

  const answer = await callLLM(messages, { temperature: 0.3, maxTokens: 4096 });
  return { answer };
}

export default { definition, handler };
