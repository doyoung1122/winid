import { encode, decode } from "gpt-tokenizer";

/**
 * 텍스트를 토큰 단위로 나눠서 chunk 리스트 반환
 * @param {string} text 원본 텍스트
 * @param {number} maxTokens 청크 최대 토큰 수
 * @param {number} overlap 청크 간 겹치는 토큰 수
 * @returns {Array<{ text: string, startTok: number, endTok: number }>}
 */
export function chunkTextTokens(text, maxTokens = 800, overlap = 100) {
  if (overlap >= maxTokens) {
    throw new Error(`overlap(${overlap}) must be < maxTokens(${maxTokens})`);
  }

  const tokens = encode(text);
  const chunks = [];

  let start = 0;
  while (start < tokens.length) {
    const end = Math.min(start + maxTokens, tokens.length);
    const chunkTokens = tokens.slice(start, end);
    const chunkText = decode(chunkTokens).trim();
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

/**
 * 섹션(헤딩) 기반 계층형 청킹
 * Docling Markdown의 ## / ### 헤딩 구조를 활용해 의미 단위로 분할.
 * 각 청크 앞에 [문서명 > 섹션명] 컨텍스트 헤더를 주입해 임베딩 품질을 높입니다.
 * 헤딩 구조가 없으면 chunkTextTokens 로 fallback.
 * @param {string} markdownText - Docling이 생성한 Markdown 텍스트
 * @param {string} docTitle - 문서 제목 (컨텍스트 헤더에 사용)
 * @param {number} maxTokens - 청크 최대 토큰 수
 * @param {number} overlapTokens - 하위 분할 시 겹침 토큰 수
 * @returns {Array<{ text: string, startTok: number, endTok: number }>}
 */
export function chunkBySections(markdownText, docTitle = "", maxTokens = 500, overlapTokens = 50) {
  const MIN_TOKENS = 80;

  // ## / ### 헤딩 경계로 분리
  const parts = markdownText
    .split(/\n(?=#{1,3} )/)
    .map((s) => s.trim())
    .filter(Boolean);

  // 헤딩 구조 없으면 일반 토큰 청킹 fallback
  if (parts.length <= 1) {
    return chunkTextTokens(markdownText, maxTokens, overlapTokens);
  }

  const sections = parts
    .map((part) => {
      const nl = part.indexOf("\n");
      if (nl === -1) return { heading: part.replace(/^#{1,3} /, "").trim(), content: "" };
      return {
        heading: part.slice(0, nl).replace(/^#{1,3} /, "").trim(),
        content: part.slice(nl + 1).trim(),
      };
    })
    .filter((s) => s.content.length > 0);

  const result = [];
  let pending = ""; // 너무 작은 섹션 병합 버퍼

  for (const { heading, content } of sections) {
    const header = docTitle ? `[${docTitle} > ${heading}]\n\n` : `[${heading}]\n\n`;
    const fullText = header + content;
    const tokenCount = encode(fullText).length;

    if (tokenCount < MIN_TOKENS) {
      // 너무 작은 섹션 → 병합 버퍼에 축적
      pending += (pending ? "\n\n" : "") + fullText;
      continue;
    }

    // 병합 버퍼 flush
    if (pending) {
      if (encode(pending).length >= MIN_TOKENS) {
        result.push(...chunkTextTokens(pending, maxTokens, overlapTokens));
      }
      pending = "";
    }

    if (tokenCount <= maxTokens) {
      result.push({ text: fullText, startTok: 0, endTok: tokenCount });
    } else {
      // 섹션이 너무 크면 토큰 단위 재분할
      result.push(...chunkTextTokens(fullText, maxTokens, overlapTokens));
    }
  }

  // 남은 버퍼 처리
  if (pending && encode(pending).length >= MIN_TOKENS) {
    result.push(...chunkTextTokens(pending, maxTokens, overlapTokens));
  }

  return result;
}

export default { chunkTextTokens, chunkBySections };
