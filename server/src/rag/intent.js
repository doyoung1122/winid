import { callLLM, withTimeout } from "../services/llm.service.js";

function getTodayStr() {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

/**
 * 질문의 인텐트 분류 + 오타/구어체 교정 + 엔티티 추출을 LLM 단일 호출로 처리
 * @param {string} question
 * @returns {Promise<{
 *   intent: "smalltalk"|"stats"|"case",
 *   query: string,
 *   entities: {
 *     region: string|null,
 *     year: number|null,
 *     building: string|null,
 *     date_ref: "어제"|"오늘"|"그제"|"YYYY-MM-DD"|null
 *   }
 * }>}
 */
export async function classifyIntent(question) {
  const todayStr = getTodayStr();
  const thisYear = todayStr.slice(0, 4);
  try {
    const answer = await withTimeout(
      callLLM(
        [
          {
            role: "system",
            content: `사용자의 질문에서 오타·구어체를 교정하고, 의도와 핵심 엔티티를 추출하세요.
JSON 형식으로만 응답하세요. 다른 텍스트는 출력하지 마세요.
오늘 날짜: ${todayStr}

의도 분류 기준:
- smalltalk: 화재와 무관한 인사·잡담
- stats: 화재 통계·건수·추이·피해 수치 조회 (날짜 포함)
- case: 화재 사례 조회·유사 사례·감식 방법론·일반 화재 지식

엔티티 추출 규칙 (해당 없으면 null):
- region: 지역명 문자열 (예: "강남구", "이천", "서울특별시")
- year: 연도 숫자 (예: 2021)
- building: 건물·장소 유형 문자열 (예: "아파트", "공장", "창고", "물류센터", "공원", "시장", "학교")
  * 특정 장소명에서 유형을 추출: "세종호수공원" → "공원", "이마트" → "마트", "한강공원" → "공원"
- region: 구체적인 장소명이 아닌 행정구역 단위만 추출 (예: "세종호수공원" → "세종", "강남구청" → "강남구")
- date_ref: 날짜 참조 — "어제" | "오늘" | "그제" | "YYYY-MM-DD" | "YYYY-MM" | "YYYY-MM-DD~YYYY-MM-DD" 형식 | null
  * "최근", "최신", "요즘", "근래", "이번 달", "이번주" 등 모호한 최신 표현 → "오늘"
  * "지난달", "저번달", "전달", "지난 달" → 오늘 기준 이전 달을 "YYYY-MM" 형식으로 반환
    예: 오늘이 2026-02-25이면 → "2026-01"
  * "지지난달", "두 달 전" → 오늘 기준 2개월 전을 "YYYY-MM" 형식으로 반환
  * "YYYY년 M월" 형식 → "YYYY-MM" 형식으로 반환 (예: "2024년 3월" → "2024-03")
  * 연도 없이 "M월 D일" 또는 "M/D" 형식 → 오늘 기준 연도(${thisYear})를 붙여 "YYYY-MM-DD" 반환
    예: "2월 23일" → "${thisYear}-02-23", "1월 5일" → "${thisYear}-01-05"
  * "지난 N일", "지난 일주일", "최근 N일" → 오늘 기준 N일 전부터 어제까지 "YYYY-MM-DD~YYYY-MM-DD" 형식
    예: 오늘이 2026-02-25이고 "지난 일주일" → "2026-02-18~2026-02-24"
  * "이번 주" → 이번 주 월요일부터 어제까지 "YYYY-MM-DD~YYYY-MM-DD" 형식

응답 예시:
{"intent":"case","query":"서울 강남구 은마아파트 화재 유사 사례","entities":{"region":"강남구","year":null,"building":"아파트","date_ref":null}}
{"intent":"stats","query":"어제의 화재 통계를 말해줘","entities":{"region":null,"year":null,"building":null,"date_ref":"어제"}}
{"intent":"stats","query":"최근 화재 통계","entities":{"region":null,"year":null,"building":null,"date_ref":"오늘"}}
{"intent":"stats","query":"2월 23일 화재 통계","entities":{"region":null,"year":null,"building":null,"date_ref":"${thisYear}-02-23"}}
{"intent":"case","query":"2021년 이천 물류센터 화재 사례","entities":{"region":"이천","year":2021,"building":"물류센터","date_ref":null}}
{"intent":"smalltalk","query":"안녕하세요","entities":{"region":null,"year":null,"building":null,"date_ref":null}}`,
          },
          { role: "user", content: question },
        ],
        { temperature: 0.0, maxTokens: 150 }
      ),
      25000,
      "intent"
    );

    const parsed = JSON.parse(answer.trim());
    const intent = ["smalltalk", "stats", "case"].includes(parsed.intent) ? parsed.intent : "case";
    const query = typeof parsed.query === "string" && parsed.query.trim() ? parsed.query.trim() : question;
    const entities = {
      region:   parsed.entities?.region   ?? null,
      year:     parsed.entities?.year     ? Number(parsed.entities.year) : null,
      building: parsed.entities?.building ?? null,
      date_ref: parsed.entities?.date_ref ?? null,
    };
    return { intent, query, entities };
  } catch (e) {
    console.error("intent check failed, defaulting to case:", e.message);
    return { intent: "case", query: question, entities: { region: null, year: null, building: null, date_ref: null } };
  }
}

export default { classifyIntent };
