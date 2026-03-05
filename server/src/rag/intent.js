import { callLLM, withTimeout } from "../services/llm.service.js";

function getDateContext() {
  const pad = (n) => String(n).padStart(2, "0");
  const d = new Date();
  const fmt  = (x) => `${x.getFullYear()}-${pad(x.getMonth() + 1)}-${pad(x.getDate())}`;
  const fmtM = (x) => `${x.getFullYear()}-${pad(x.getMonth() + 1)}`;

  const yesterday   = new Date(d); yesterday.setDate(d.getDate() - 1);
  const twoDaysAgo  = new Date(d); twoDaysAgo.setDate(d.getDate() - 2);
  const threeDaysAgo = new Date(d); threeDaysAgo.setDate(d.getDate() - 3);

  // 이번 주: 월요일~오늘
  const weekStart = new Date(d);
  weekStart.setDate(d.getDate() - ((d.getDay() + 6) % 7)); // 월요일
  // 지난 주: 저번 월요일~저번 일요일
  const lastWeekEnd   = new Date(weekStart); lastWeekEnd.setDate(weekStart.getDate() - 1);
  const lastWeekStart = new Date(weekStart); lastWeekStart.setDate(weekStart.getDate() - 7);

  const lastMonthStart = new Date(d.getFullYear(), d.getMonth() - 1, 1);

  // 최근 N년간 / N개월간 범위 계산
  const recentYears = [1, 2, 3, 5, 10].map(n => {
    const start = new Date(d.getFullYear() - n, 0, 1);
    return `최근${n}년간=${fmt(start)}~${fmt(d)}`;
  });
  const recentMonths = [1, 2, 3, 5, 6, 12].map(n => {
    const start = new Date(d.getFullYear(), d.getMonth() - n, 1);
    return `최근${n}개월간=${fmt(start)}~${fmt(d)}`;
  });

  const dateLookup = [
    `오늘=${fmt(d)}`,
    `어제=${fmt(yesterday)}`,
    `그저께=${fmt(twoDaysAgo)}`,
    `3일전=${fmt(threeDaysAgo)}`,
    `이번주=${fmt(weekStart)}~${fmt(d)}`,
    `지난주=${fmt(lastWeekStart)}~${fmt(lastWeekEnd)}`,
    `이번달=${fmtM(d)}`,
    `지난달=${fmtM(lastMonthStart)}`,
    ...recentYears,
    ...recentMonths,
  ].join(", ");

  return {
    today:    fmt(d),
    thisYear: String(d.getFullYear()),
    dateLookup,
    exYesterday: fmt(yesterday),
    exPrevMonth: fmtM(lastMonthStart),
  };
}

/**
 * 질문의 인텐트 분류 + 오타/구어체 교정 + 엔티티 추출을 LLM 단일 호출로 처리
 * @param {string} question
 * @param {Array<{user:string, assistant:string}>} [history] - 최근 대화 턴 (컨텍스트 파악용)
 * @returns {Promise<{
 *   intent: "smalltalk"|"stats"|"case",
 *   query: string,
 *   entities: {
 *     region: string|null,    // 시도(광역) 단위
 *     district: string|null,  // 시군구 단위
 *     year: number|null,
 *     building: string|null,
 *     date_ref: string|null   // YYYY-MM-DD | YYYY-MM | YYYY-MM-DD~YYYY-MM-DD
 *   }
 * }>}
 */
export async function classifyIntent(question, history = []) {
  const { today, thisYear, dateLookup, exYesterday, exPrevMonth } = getDateContext();

  // 직전 1~2턴을 컨텍스트 힌트로 포함 (대화 흐름 파악용)
  const recentTurns = history.slice(-2);
  const historyHint = recentTurns.length > 0
    ? `\n\n[직전 대화 (인텐트 분류 참고용)]:\n` +
      recentTurns.map(t =>
        `사용자: ${t.user ?? ""}\n어시스턴트: ${(t.assistant ?? "").slice(0, 80)}`
      ).join("\n")
    : "";

  try {
    const answer = await withTimeout(
      callLLM(
        [
          {
            role: "system",
            content: `사용자의 질문에서 오타·구어체를 교정하고, 의도와 핵심 엔티티를 추출하세요.
JSON 형식으로만 응답하세요. 다른 텍스트는 출력하지 마세요.
오늘 날짜: ${today}

의도 분류 기준:
- smalltalk  : 화재와 무관한 인사·잡담
- stats      : 화재 통계·건수·추이·피해 수치 등 수치 데이터 조회 (데이터베이스 조회)
- case       : 특정 화재 사례 조회·유사 사례 검색 (데이터베이스 조회)
- document   : 화재조사 절차·방법론·감식 기법·법령·매뉴얼·일반 화재 지식 질문 (업로드 문서 검색)

[case vs document 구분]
- case    : "~화재 사례", "유사 사례", "~에서 발생한 화재" → 특정 사건 DB 조회
- document: "어떻게 하나요", "절차", "방법", "기준", "원칙", "매뉴얼", "감식", "조사 방법" → 문서/지식 검색

[중요] 직전 대화가 제공된 경우 대화 흐름을 반드시 고려하세요.
- 이전 턴이 화재 관련이었다면 짧은 후속 질문도 적절한 intent로 분류

엔티티 추출 규칙 (해당 없으면 null):
- region: 시도(광역) 단위 지역명. 해당 없으면 null. 반드시 공식 전체 명칭 사용
  * 광역시·특별시·도 단위만 region으로 분류. 일반 시(市)·군(郡)·구(區)는 district
  * 약칭→정식명: 서울→서울특별시, 부산→부산광역시, 대구→대구광역시, 인천→인천광역시,
    광주→광주광역시, 대전→대전광역시, 울산→울산광역시, 세종→세종특별자치시,
    경기→경기도, 강원→강원도, 충북→충청북도, 충남→충청남도,
    전북→전라북도, 전남→전라남도, 경북→경상북도, 경남→경상남도, 제주→제주특별자치도
  * 수원·안산·안양·용인·성남·고양·창원·청주·천안·전주·포항 등 일반 시(市)는 절대 region 사용 금지 → district로 분류
  * 시도+구/시/군 동시 언급 → region=시도(정식명), district=시군구
  * 시도만 언급 → region=시도(정식명), district=null
  * 구/시/군만 언급 → region=null, district=해당값
- district: 시군구 단위 지역명. 해당 없으면 null (예: "해운대구", "이천시", "마포구")
  * 행정 단위 접미사 유지: "이천" → "이천시", "강남" → "강남구", "여수" → "여수시"
- year: 연도 숫자 (예: 2021)
- building: 건물·장소 유형 (예: "아파트", "공장", "물류센터")
  * 특정 장소명에서 유형 추출: "세종호수공원" → "공원", "이마트" → "마트"
- date_ref: 아래 날짜 참고 테이블을 먼저 확인하고, "YYYY-MM-DD" | "YYYY-MM" | "YYYY-MM-DD~YYYY-MM-DD" | null 반환
  * 테이블에 있는 표현은 해당 값 그대로 사용 (직접 계산 불필요)
  * 테이블에 없는 표현(예: "5일 전", "3월 15일")만 직접 계산
  * 연도 없이 "M월 D일" → ${thisYear} 기준으로 변환
  * "최근", "요즘" 등 모호한 최신 표현 → 오늘 날짜 "${today}"
  * "최근 N년간", "최근 N년" → 위 테이블의 해당 항목 그대로 사용 (예: 최근5년간)
  * "최근 N개월간", "최근 N개월" → 위 테이블의 해당 항목 그대로 사용

[날짜 참고 테이블 (요청 시점 기준 계산 완료)]
${dateLookup}

응답 예시 (오늘이 ${today}인 경우):
{"intent":"stats","query":"어제 화재 통계","entities":{"region":null,"district":null,"year":null,"building":null,"date_ref":"${exYesterday}"}}
{"intent":"stats","query":"2월 23일 화재 통계","entities":{"region":null,"district":null,"year":null,"building":null,"date_ref":"${thisYear}-02-23"}}
{"intent":"stats","query":"지난달 서울 화재 건수","entities":{"region":"서울특별시","district":null,"year":null,"building":null,"date_ref":"${exPrevMonth}"}}
{"intent":"stats","query":"2012년 충남 화재 통계","entities":{"region":"충청남도","district":null,"year":2012,"building":null,"date_ref":null}}
{"intent":"case","query":"부산 해운대구 아파트 화재 사례","entities":{"region":"부산광역시","district":"해운대구","year":null,"building":"아파트","date_ref":null}}
{"intent":"case","query":"2021년 이천 물류센터 화재 사례","entities":{"region":null,"district":"이천시","year":2021,"building":"물류센터","date_ref":null}}
{"intent":"document","query":"화재 발화점 감식 방법","entities":{"region":null,"district":null,"year":null,"building":null,"date_ref":null}}
{"intent":"document","query":"화재조사 현장 보존 절차","entities":{"region":null,"district":null,"year":null,"building":null,"date_ref":null}}
{"intent":"document","query":"화재안전특별조사 운영 매뉴얼","entities":{"region":null,"district":null,"year":null,"building":null,"date_ref":null}}
{"intent":"smalltalk","query":"안녕하세요","entities":{"region":null,"district":null,"year":null,"building":null,"date_ref":null}}

[직전 대화 있을 때 예시]
직전: 사용자="강남구 아파트 화재 원인 분석해줘" / 어시스턴트="전기적 요인이..."
현재 질문: "방금 내용 요약해줘"
→ {"intent":"case","query":"강남구 아파트 화재 원인 분석 요약","entities":{"region":null,"district":"강남구","year":null,"building":"아파트","date_ref":null}}${historyHint}`,
          },
          { role: "user", content: question },
        ],
        { temperature: 0.0, maxTokens: 200 }
      ),
      25000,
      "intent"
    );

    const parsed = JSON.parse(answer.trim());
    const intent = ["smalltalk", "stats", "case", "document"].includes(parsed.intent) ? parsed.intent : "case";
    const query = typeof parsed.query === "string" && parsed.query.trim() ? parsed.query.trim() : question;
    const entities = {
      region:   parsed.entities?.region    ?? null,
      district: parsed.entities?.district  ?? null,
      year:     parsed.entities?.year      ? Number(parsed.entities.year) : null,
      building: parsed.entities?.building  ?? null,
      date_ref: parsed.entities?.date_ref  ?? null,
    };
    return { intent, query, entities };
  } catch (e) {
    console.error("intent check failed, defaulting to case:", e.message);
    return { intent: "case", query: question, entities: { region: null, district: null, year: null, building: null, date_ref: null } };
  }
}

