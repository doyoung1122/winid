import { LLM_URL, LLM_MODEL } from "../config/env.js";

/**
 * Call LLM with chat completion API
 * @param {Array<{role: string, content: string}>} messages - Chat messages
 * @param {Object} options - LLM options
 * @returns {Promise<string>} LLM response content
 */
export async function callLLM(messages, options = {}) {
  const { temperature = 0.7, maxTokens = 2048, stream = false } = options;

  const response = await fetch(`${LLM_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages,
      temperature,
      max_tokens: maxTokens,
      stream,
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
  const { temperature = 0.7, maxTokens = 2048, top_p = 0.9 } = options;

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
 * @param {ReadableStream} readable - SSE readable stream
 * @returns {Promise<string>} Accumulated text
 */
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
          } catch (e) {
            // Ignore parse errors
          }
        }
      }
    });
    readable.on("end", () => resolve(acc));
    readable.on("error", (err) => reject(err));
  });
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
  checkLLMHealth,
  withTimeout,
};
