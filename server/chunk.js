const { encode, decode} = require("gpt-tokenizer");

/**
 * 텍스트를 토큰 단위로 나눠서 chunk 리스트 반환
 * @param {string} text 원본 텍스트
 * @param {number} maxTokens 청크 최대 토큰 수
 * @param {number} overlap 청크 간 겹치는 토큰 수
 * @returns {Array<{ text: string, startTok: number, endTok: number }>}
 */

function chunkTextTokens(text, maxTokens = 800, overlap = 100) {
  if (overlap >= maxTokens) {
    throw new Error(`overlap(${overlap}) must be < maxTokens(${maxTokens})`);
  }

  const tokens = encode(text);
  const chunks = [];

  let start = 0;
  while (start < tokens.length) {
    const end = Math.min(start + maxTokens, tokens.length);
    const chunkTokens = tokens.slice(start, end);
    const chunkText = decodeTokens(chunkTokens).trim();
    if (chunkText.length > 0) {
      chunks.push({
        text: chunkText,
        startTok: start,
        endTok: end,
      });
    }
    start += maxTokens - overlap;
  }
  return chunks;
}

// 디코딩 함수
function decodeTokens(tokens) {
  return decode(tokens);
}

module.exports = { chunkTextTokens };