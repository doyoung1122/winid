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
  const { temperature = 0.7, maxTokens = 2048, top_p = 0.9, enableThinking = false } = options;

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
  const { temperature = 0.7, maxTokens = 2048, top_p = 0.9, enableThinking = false } = options;

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

export default {
  callLLM,
  callLLMStream,
  callLLMStreamTokens,
  checkLLMHealth,
  withTimeout,
};
