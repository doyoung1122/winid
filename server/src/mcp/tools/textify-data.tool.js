import { callLLM } from "../../services/llm.service.js";
import { SYSTEM_TEXTIFIER } from "../../rag/prompts.js";

export const definition = {
  name: "textify_data",
  description:
    "JSON 형태의 화재 통계 데이터를 자연어 문장으로 변환합니다. 임베딩 전 텍스트화에 사용합니다.",
  inputSchema: {
    type: "object",
    properties: {
      data: {
        type: "object",
        description: "변환할 JSON 데이터",
      },
      dataType: {
        type: "string",
        enum: ["fire_stats", "incident_report", "inspection_result"],
        description: "데이터 유형",
      },
    },
    required: ["data"],
  },
};

export async function handler({ data, dataType }) {
  const dataStr = JSON.stringify(data, null, 2);
  const typeHint = dataType ? `데이터 유형: ${dataType}\n` : "";

  const messages = [
    { role: "system", content: SYSTEM_TEXTIFIER },
    {
      role: "user",
      content: `${typeHint}다음 데이터를 자연어 문장으로 변환하세요:\n\n${dataStr}`,
    },
  ];

  const text = await callLLM(messages, { temperature: 0.1, maxTokens: 1024 });
  return { text };
}

export default { definition, handler };
