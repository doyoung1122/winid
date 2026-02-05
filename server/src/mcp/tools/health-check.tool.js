import { checkLLMHealth } from "../../services/llm.service.js";
import { checkEmbeddingHealth } from "../../services/embedding.service.js";

export const definition = {
  name: "health_check",
  description: "LLM 및 Embedding 서버 상태를 확인합니다.",
  inputSchema: {
    type: "object",
    properties: {},
  },
};

export async function handler() {
  const llmHealth = await checkLLMHealth();
  const embHealth = await checkEmbeddingHealth();

  return {
    llm: llmHealth.ok,
    llmStatus: llmHealth.status,
    embedding: embHealth.ok,
    embeddingStatus: embHealth.status,
  };
}

export default { definition, handler };
