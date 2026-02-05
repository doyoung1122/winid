import { getEmbedding } from "../../services/embedding.service.js";
import { EMB_MODEL } from "../../config/env.js";

export const definition = {
  name: "get_embedding",
  description: "텍스트의 임베딩 벡터를 생성합니다 (bge-m3, 1024차원).",
  inputSchema: {
    type: "object",
    properties: {
      text: {
        type: "string",
        description: "임베딩할 텍스트",
      },
    },
    required: ["text"],
  },
};

export async function handler({ text }) {
  const embedding = await getEmbedding(text);
  return {
    embedding,
    dimension: embedding.length,
    model: EMB_MODEL,
  };
}

export default { definition, handler };
