import { readFile } from "fs/promises";
import { LLM_URL, LLM_MODEL } from "../config/env.js";

/**
 * Call LLM with chat completion API
 * @param {Array<{role: string, content: string}>} messages - Chat messages
 * @param {Object} options - LLM options
 * @returns {Promise<string>} LLM response content
 */
export async function callLLM(messages, options = {}) {
  const { temperature = 0.7, maxTokens = 2048, stream = false, enableThinking = false } = options;

  const response = await fetch(`${LLM_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages,
      temperature,
      max_tokens: maxTokens,
      stream,
      chat_template_kwargs: { enable_thinking: enableThinking },
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`LLM request failed: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return data.choices[0]?.message?.content || "";
}

/**
 * Call LLM with streaming response
 * @param {Array<{role: string, content: string}>} messages - Chat messages
 * @param {Object} options - LLM options
 * @returns {Promise<string>} Accumulated response text
 */
export async function callLLMStream(messages, options = {}) {
  const { temperature = 0.7, maxTokens = 2048, top_p = 0.9, enableThinking = false, repetitionPenalty = 1.15 } = options;

  const response = await fetch(`${LLM_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages,
      temperature,
      max_tokens: maxTokens,
      top_p,
      repetition_penalty: repetitionPenalty,
      stream: true,
      chat_template_kwargs: { enable_thinking: enableThinking },
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`LLM call failed: ${response.status} ${err}`);
  }

  const text = await readSSEToText(response.body);
  return text.trim();
}

/**
 * Read SSE stream and accumulate text
 * @param {ReadableStream} body - WHATWG ReadableStream from fetch response
 * @returns {Promise<string>} Accumulated text
 */
async function readSSEToText(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let acc = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    for (const line of chunk.split(/\r?\n/)) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6);
      if (payload === "[DONE]") continue;
      try {
        const content = JSON.parse(payload).choices?.[0]?.delta?.content || "";
        acc += content;
      } catch (_) {}
    }
  }
  return acc;
}

/**
 * Call LLM with streaming, invoking onToken for each token
 * @param {Array<{role: string, content: string}>} messages
 * @param {Object} options
 * @param {(token: string) => void} onToken
 * @returns {Promise<string>} Accumulated response text
 */
export async function callLLMStreamTokens(messages, options = {}, onToken) {
  const { temperature = 0.7, maxTokens = 2048, top_p = 0.9, enableThinking = false, repetitionPenalty = 1.15 } = options;

  const response = await fetch(`${LLM_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages,
      temperature,
      max_tokens: maxTokens,
      top_p,
      repetition_penalty: repetitionPenalty,
      stream: true,
      chat_template_kwargs: { enable_thinking: enableThinking },
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`LLM call failed: ${response.status} ${err}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let acc = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    for (const line of chunk.split(/\r?\n/)) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6);
      if (payload === "[DONE]") continue;
      try {
        const content = JSON.parse(payload).choices?.[0]?.delta?.content || "";
        if (content) {
          acc += content;
          onToken(content);
        }
      } catch (_) {}
    }
  }
  return acc;
}

/**
 * Health check for LLM service
 * @returns {Promise<{ok: boolean, status?: string}>}
 */
export async function checkLLMHealth() {
  try {
    const res = await fetch(`${LLM_URL}/health`, { method: "GET" });
    return {
      ok: res.ok,
      status: res.ok ? "healthy" : `status ${res.status}`,
    };
  } catch (e) {
    return {
      ok: false,
      status: `error: ${e.message}`,
    };
  }
}

/**
 * Helper function with timeout
 * @param {Promise} promise - Promise to wrap
 * @param {number} ms - Timeout in milliseconds
 * @param {string} tag - Tag for error message
 * @returns {Promise} Promise with timeout
 */
export function withTimeout(promise, ms = 30000, tag = "req") {
  return Promise.race([
    promise,
    new Promise((_, rej) => setTimeout(() => rej(new Error(`${tag} timeout ${ms}ms`)), ms)),
  ]);
}

/**
 * Qwen3-VL로 이미지를 설명하는 함수 (화재조사 문서 이미지 전용)
 * @param {string} imagePath - 절대 경로
 * @param {string} docContext - 문서 제목/맥락 (프롬프트 보강용)
 * @returns {Promise<string>} 한국어 설명 텍스트
 */
export async function describeImage(imagePath, docContext = "") {
  const imageBuffer = await readFile(imagePath);
  const b64 = imageBuffer.toString("base64");
  const ext = imagePath.toLowerCase().endsWith(".png") ? "png" : "jpeg";

  const prompt = docContext
    ? `이 이미지는 화재조사 문서("${docContext}")에서 추출된 것입니다. 표·도표·사진·도면 중 어떤 형태인지 파악하고, 화재조사에 관련된 핵심 내용을 간결하게 설명해주세요. (4~6문장)`
    : `이 이미지의 유형(표, 도표, 사진, 도면 등)을 파악하고 화재조사와 관련된 핵심 내용을 설명해주세요.`;

  const response = await fetch(`${LLM_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: `data:image/${ext};base64,${b64}` } },
            { type: "text", text: prompt },
          ],
        },
      ],
      max_tokens: 300,
      temperature: 0.2,
      chat_template_kwargs: { enable_thinking: false },
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`VLM describe failed: ${response.status} - ${err}`);
  }

  const data = await response.json();
  return data.choices[0]?.message?.content?.trim() || "";
}

export default {
  callLLM,
  callLLMStream,
  callLLMStreamTokens,
  checkLLMHealth,
  withTimeout,
  describeImage,
};
