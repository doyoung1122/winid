import { callLLM, withTimeout } from "../services/llm.service.js";
import { pool } from "../../../db/mysql.js";

// =========================
// 스키마 정의
// =========================

const STATS_SCHEMA = `
[테이블: vfs_data_api] 날짜별 화재 통계 (2008-01-01 ~ 현재)
컬럼:
  fire_date         DATE          -- 날짜
  stat_type         VARCHAR(50)   -- 분류 기준 (아래 목록 참조)
  region            VARCHAR(50)   -- 지역 ('전국' 또는 시도명)
  category_main     VARCHAR(100)  -- 대분류
  category_sub      VARCHAR(100)  -- 소분류 (없으면 '')
  fire_count        INT           -- 화재 건수
  death_count       INT           -- 사망자
  injury_count      INT           -- 부상자
  life_damage_count INT           -- 인명피해 합계
  damage_amount     BIGINT        -- 재산피해 (천원)

stat_type 값 & region 규칙:
  '건물구조'        → region='전국'만
  '발화열원'        → region='전국'만
  '발화요인'        → region='전국'만
  '발화장소'        → region='전국'만
  '발화지점'        → region='전국'만
  '선박항공기'      → region='전국'만
  '임야'            → region='전국'만
  '차량발화지점'    → region='전국'만
  '최초착화물'      → region='전국'만
  '화재장소(전국)'  → region='전국'만 (장소별 전국 집계)
  '시도별'          → region=시도명 (시도별 인명+재산피해 집계)
  '화재장소(시도)'  → region=시도명 × 장소 교차 집계

시도명: '서울특별시','부산광역시','대구광역시','인천광역시','광주광역시','대전광역시','울산광역시','경기도','강원도','충청북도','충청남도','전라북도','전라남도','경상북도','경상남도','제주특별자치도'

날짜 필터:
  특정일: WHERE fire_date='YYYY-MM-DD'
  특정월: WHERE YEAR(fire_date)=YYYY AND MONTH(fire_date)=MM
  특정년: WHERE YEAR(fire_date)=YYYY
  범위:   WHERE fire_date BETWEEN 'start' AND 'end'

시도별 화재 건수: stat_type='시도별', region=해당 시도명 (예: '경기도')
전국 총계:        stat_type='시도별' (region 조건 없이 모든 시도 SUM = 전국 합계)
전국 분류 통계:   region='전국'만인 stat_type들 (발화요인, 발화장소, 건물구조 등)

[필수 규칙] vfs_data_api 쿼리는 반드시 stat_type 조건을 포함해야 합니다.
stat_type 없이 조회하면 9개 stat_type이 모두 합산되어 수치가 최대 9배 부풀어 오릅니다.

[stat_type='시도별' 사용 규칙]
- region='전국' 행은 존재하지 않습니다. "전국" 통계 질문에 region='전국' 사용 금지.
- 전국 합계: stat_type='시도별' + region 조건 없이 → 16개 시도 합산 = 전국 합계
  예) SELECT SUM(fire_count) AS 화재건수 FROM vfs_data_api WHERE stat_type='시도별' AND YEAR(fire_date)=2024
- 특정 시도: stat_type='시도별' AND region='경기도'

[중요 제약] vfs_data_api는 건물유형(아파트, 공장 등) 필터가 없습니다.
"아파트", "공장", "주택" 등 건물유형이 질문에 포함되더라도
stat_type='시도별' 또는 '화재장소(시도)'로 조회하고,
건물유형 조건은 SQL에 추가하지 마세요.

[집계 쿼리 규칙 - 중요]
- "N년 전체", "N년 합계", "총 몇 건" 등 기간 합산 질문 → 반드시 SUM() 사용
  예) SELECT SUM(fire_count) as 화재건수, SUM(death_count) as 사망자, SUM(injury_count) as 부상자, SUM(damage_amount) as 재산피해
      FROM vfs_data_api WHERE YEAR(fire_date)=2010 AND region='충청남도' AND stat_type='시도별'
- 전국 연도별 추이: SELECT YEAR(fire_date) AS 연도, SUM(fire_count) AS 화재건수, SUM(death_count) AS 사망자, SUM(injury_count) AS 부상자, SUM(damage_amount) AS 재산피해 FROM vfs_data_api WHERE stat_type='시도별' GROUP BY YEAR(fire_date) ORDER BY 연도 DESC
- 전국 월별 추이: SELECT YEAR(fire_date) AS 연도, MONTH(fire_date) AS 월, SUM(fire_count) AS 화재건수, SUM(death_count) AS 사망자, SUM(injury_count) AS 부상자, SUM(damage_amount) AS 재산피해 FROM vfs_data_api WHERE stat_type='시도별' GROUP BY YEAR(fire_date), MONTH(fire_date) ORDER BY 연도, 월
- stat_type별 분류 현황 → GROUP BY category_main + SUM()
  예) SELECT category_main, SUM(fire_count) as 화재건수 FROM vfs_data_api
      WHERE YEAR(fire_date)=2010 AND stat_type='발화요인' AND region='전국'
      GROUP BY category_main ORDER BY 화재건수 DESC
- 개별 날짜 상세 조회만 LIMIT 사용 (기간 합산에 LIMIT 사용 금지)`.trim();

const CASE_SCHEMA = `
[테이블: vfs_data_csv] 개별 화재 사례 (2019~2023, 총 194,105건)
컬럼:
  id INT, report_no VARCHAR(20)
  year SMALLINT, month TINYINT, day TINYINT, hour TINYINT (NULL 가능)
  fire_type    VARCHAR(50)  -- 화재 대분류
  building_type      VARCHAR(50)  -- 건물 구조 방식 (조립식/한식(옥)/양식(옥)/절충식 등, 용도 아님)
  building_structure VARCHAR(50)  -- 건물 구조 재질
  location_main VARCHAR(100) -- 발화장소 대분류
  location_mid  VARCHAR(100) -- 발화장소 중분류
  location_sub  VARCHAR(100) -- 발화장소 소분류
  cause_main VARCHAR(100), cause_sub VARCHAR(100) -- 발화요인
  heat_source  VARCHAR(100) -- 발화열원
  device_main VARCHAR(100), device_sub VARCHAR(100) -- 발화기기
  first_object_main VARCHAR(100), first_object_sub VARCHAR(100) -- 최초착화물
  death_count INT, injury_count INT -- 인명피해
  damage_amount BIGINT -- 재산피해
  suppression_time FLOAT -- 진압시간(초)
  region VARCHAR(50)   -- 시도
  district VARCHAR(50) -- 시군구
  temperature FLOAT, wind_speed VARCHAR(20), humidity FLOAT -- 기상

[fire_type 값]
  '건축,구조물' | '기타(쓰레기 화재등)' | '선박,항공기' | '위험물,가스제조소등' | '임야' | '자동차,철도차량'

[location_main > location_mid 전체 목록]
  주거 > 공동주택(아파트·연립·다세대) | 주거 > 단독주택 | 주거 > 기타주택
  산업시설 > 공장시설 | 산업시설 > 창고시설 | 산업시설 > 작업장 | 산업시설 > 동식물시설
  생활서비스 > 음식점 | 생활서비스 > 일상서비스 | 생활서비스 > 오락시설 | 생활서비스 > 위락시설
  판매,업무시설 > 판매시설 | 판매,업무시설 > 일반업무 | 판매,업무시설 > 숙박시설 | 판매,업무시설 > 공공기관
  교육시설 > 학교 | 교육시설 > 연구,학원
  의료,복지시설 > 의료시설 | 의료,복지시설 > 노유자시설 | 의료,복지시설 > 건강시설
  집합시설 > 종교시설 | 집합시설 > 운동시설 | 집합시설 > 공연장
  자동차,철도차량 > 자동차 | 자동차,철도차량 > 철도차량
  임야 > 산불 | 임야 > 들불
  기타 > 도로 | 기타 > 야외

[주요 장소 매핑 - 반드시 location 컬럼 사용]
  아파트/연립/다세대  → location_main='주거'    AND location_mid='공동주택'
  단독주택/주택       → location_main='주거'    AND location_mid='단독주택'
  공장                → location_main='산업시설' AND location_mid='공장시설'
  창고/물류           → location_main='산업시설' AND location_mid='창고시설'
  음식점/식당         → location_main='생활서비스' AND location_mid='음식점'
  학교                → location_main='교육시설' AND location_mid='학교'
  병원                → location_main='의료,복지시설' AND location_mid='의료시설'
  자동차              → location_main='자동차,철도차량' AND location_mid='자동차'

[지역 값]
  '서울특별시','부산광역시','대구광역시','인천광역시','광주광역시','대전광역시','울산광역시',
  '경기도','강원도','충청북도','충청남도','전라북도','전라남도','경상북도','경상남도','제주특별자치도'

[쿼리 가이드]
  사례 건수/집계: COUNT(*), GROUP BY 사용
  개별 사례 조회: LIMIT 3, ORDER BY damage_amount DESC, death_count DESC 권장
  건물용도 필터:  location_main, location_mid 사용 (building_type은 구조 방식)
  발화요인 필터:  cause_main='부주의'

[district 값 규칙]
  district는 반드시 행정 단위 접미사 포함:
  시: '이천시','성남시','화성시','용인시' 등
  군: '양평군','가평군' 등
  구: '강남구','강서구' 등
  예: 사용자가 "이천"이라고 하면 → district='이천시'
  예: 사용자가 "강남"이라고 하면 → district='강남구'

  [중요] 시(市) 단위 도시는 district 컬럼에 '수원시 영통구'처럼 구 단위까지 포함될 수 있습니다.
  특히 고양시·성남시·수원시·안산시·안양시·용인시·전주시·창원시·천안시·청주시·포항시는
  반드시 district LIKE '도시명%' 형태로 조회하세요.
  그 외 시(市)도 안전하게 LIKE '도시명%' 사용을 권장합니다.
  군(郡)·구(區)는 = 그대로 사용 가능합니다.`.trim();

// =========================
// SQL 생성
// =========================

export async function generateSQL(question, intent, entities) {
  const schema = intent === "stats" ? STATS_SCHEMA : CASE_SCHEMA;

  const hints = [];
  if (intent === "stats") {
    if (entities.date_ref) {
      // date_ref를 SQL WHERE 조건으로 직접 변환 (LLM이 날짜 계산하지 않도록)
      const dr = entities.date_ref;
      let dateCond;
      if (dr.includes("~")) {
        const [s, e] = dr.split("~");
        dateCond = `fire_date BETWEEN '${s}' AND '${e}'`;
      } else if (dr.length === 7) {
        dateCond = `YEAR(fire_date)=${dr.slice(0,4)} AND MONTH(fire_date)=${parseInt(dr.slice(5,7))}`;
      } else {
        dateCond = `fire_date='${dr}'`;
      }
      hints.push(`날짜 WHERE 조건 (반드시 이 조건을 그대로 사용): ${dateCond}`);
    } else {
      hints.push(`날짜 조건 없음 → 연도별 집계로 전체 추이 반환. 전국 추이 예시: SELECT YEAR(fire_date) AS 연도, SUM(fire_count) AS 화재건수, SUM(death_count) AS 사망자, SUM(injury_count) AS 부상자, SUM(damage_amount) AS 재산피해 FROM vfs_data_api WHERE stat_type='시도별' GROUP BY YEAR(fire_date) ORDER BY 연도 DESC (특정 시도라면 AND region='시도명' 추가)`);
    }
  }
  // 사례 쿼리: "오늘/어제" 같은 날짜는 현재 사건 발생 시점이며 유사 사례 검색 필터로 쓰지 않음
  // entities.year는 "2021년 이천 화재" 같은 명시적 연도는 필터로 사용
  if (entities.region)   hints.push(`시도: "${entities.region}"`);
  if (entities.district) hints.push(`시군구: "${entities.district}"`);
  if (entities.year && intent === "stats") hints.push(`연도: ${entities.year} → 반드시 stat_type='시도별' 포함. 전국 합산: WHERE stat_type='시도별' AND YEAR(fire_date)=${entities.year}. 월별 요약 시: GROUP BY MONTH(fire_date). 연도 합산 시: SUM() 집계 (LIMIT 없이)`);
  if (entities.year && intent === "case")  hints.push(`연도: ${entities.year}`);
  if (entities.building) hints.push(`건물/장소 유형: "${entities.building}"`);
  const hintStr = hints.length > 0 ? `\n[추출된 조건]\n${hints.join("\n")}\n` : "";

  const caseExtraRules = intent === "case"
    ? `- vfs_data_csv는 2019~2023년 데이터만 존재. 2024년 이후 연도 필터 절대 추가 금지\n- "오늘", "어제", "최근" 등 현재 날짜 표현은 유사 사례 검색 필터로 사용하지 않음\n`
    : "";

  const sysPmt = `당신은 화재 데이터베이스 SQL 전문가입니다. 사용자 질문에 맞는 MySQL SELECT SQL을 생성하세요.

${schema}

[생성 규칙]
- SELECT 쿼리만 허용
- 집계 없는 개별 행 조회는 LIMIT 3
- 한국어 컬럼값은 위 목록의 정확한 문자열 사용
${caseExtraRules}- 반드시 JSON 형식으로만 응답: {"sql":"...", "note":"..."}`;

  const raw = await withTimeout(
    callLLM(
      [
        { role: "system", content: sysPmt },
        { role: "user", content: `${hintStr}\n질문: ${question}` },
      ],
      { temperature: 0.0, maxTokens: 500 }
    ),
    25000,
    "sql-gen"
  );

  const m = raw.match(/\{[\s\S]*?\}/);
  if (!m) throw new Error("SQL 응답 파싱 실패");

  const { sql } = JSON.parse(m[0]);
  if (!sql) throw new Error("SQL 필드 없음");
  const { sql: finalSql, note, isLimited } = validateSQL(sql.trim());
  return { sql: finalSql, note, isLimited };
}

const ALLOWED_TABLES = new Set(["vfs_data_api", "vfs_data_csv"]);

function validateSQL(sql) {
  // 끝의 세미콜론 제거
  let clean = sql.replace(/;+\s*$/, "").trim();
  if (!/^SELECT\s/i.test(clean)) throw new Error("허용되지 않은 접근입니다");
  if (/\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|EXEC)\b/i.test(clean)) {
    throw new Error("허용되지 않은 접근입니다");
  }
  if (/\bUNION\b/i.test(clean)) throw new Error("허용되지 않은 접근입니다");
  // 허용 테이블 외 접근 차단
  const tables = [...clean.matchAll(/\bFROM\s+(\w+)/gi)].map(m => m[1].toLowerCase());
  const subTables = [...clean.matchAll(/\bJOIN\s+(\w+)/gi)].map(m => m[1].toLowerCase());
  for (const tbl of [...tables, ...subTables]) {
    if (!ALLOWED_TABLES.has(tbl)) throw new Error("허용되지 않은 접근입니다");
  }
  const hasAgg = /\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b/i.test(clean);
  const isCaseTbl = /FROM\s+vfs_data_csv\b/i.test(clean);

  // vfs_data_api: stat_type 없으면 기본값 '시도별' 자동 주입 (중복 집계 방지)
  if (!isCaseTbl && !/\bstat_type\s*=/i.test(clean)) {
    if (/\bWHERE\b/i.test(clean)) {
      clean = clean.replace(/\bWHERE\b/i, "WHERE stat_type='시도별' AND");
    } else {
      clean = clean.replace(/\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b/i, "WHERE stat_type='시도별' $1");
      if (!/\bWHERE\b/i.test(clean)) clean += " WHERE stat_type='시도별'";
    }
  }

  // vfs_data_api: stat_type과 region 조합 교정
  const NATIONAL_ONLY = ["건물구조","발화열원","발화요인","발화장소","발화지점",
                         "선박항공기","임야","차량발화지점","최초착화물","화재장소(전국)"];
  let nationalNote = "";
  if (!isCaseTbl) {
    const stMatch = clean.match(/stat_type\s*=\s*'([^']+)'/i);
    const regionMatch = clean.match(/region\s*=\s*'([^']+)'/i);
    if (stMatch) {
      // 시도별: region='전국'은 존재하지 않으므로 region 조건 제거
      if (stMatch[1] === "시도별" && regionMatch && regionMatch[1] === "전국") {
        clean = clean.replace(/\bAND\s+region\s*=\s*'전국'/gi, "")
                     .replace(/\bregion\s*=\s*'전국'\s+AND\b/gi, "")
                     .replace(/\bWHERE\s+region\s*=\s*'전국'\b/gi, "WHERE 1=1")
                     .trim();
        nationalNote = `[안내: 전국 합계는 시도별 데이터를 합산하여 계산됩니다]\n`;
      }
      // 전국 전용 stat_type에 시도 region이 있으면 '전국'으로 교정
      else if (NATIONAL_ONLY.includes(stMatch[1]) && regionMatch && regionMatch[1] !== "전국") {
        clean = clean.replace(/AND\s+region\s*=\s*'[^']+'/gi, "AND region='전국'")
                     .replace(/region\s*=\s*'[^']+'\s+AND/gi, "region='전국' AND");
        nationalNote = `[안내: '${stMatch[1]}' 통계는 지역별 분류가 없어 전국 기준으로 제공됩니다]\n`;
      }
    }
  }

  // vfs_data_csv: 전북특별자치도 명칭 교정 (vfs_data_api는 전라북도 유지)
  if (isCaseTbl) {
    clean = clean.replace(/region\s*=\s*'전라북도'/g, "region='전북특별자치도'");
  }

  // vfs_data_csv: '시'로 끝나는 district는 구 단위 포함 가능하므로 LIKE 자동 적용
  if (isCaseTbl) {
    clean = clean.replace(/district\s*=\s*'([^']+시)'/gi, "district LIKE '$1%'");
  }

  // vfs_data_csv는 2019~2023만 존재: 2024년 이후 year 필터 자동 제거
  if (isCaseTbl) {
    const yearMatch = clean.match(/\byear\s*=\s*(\d{4})\b/i);
    if (yearMatch && Number(yearMatch[1]) > 2023) {
      clean = clean.replace(/\bAND\s+year\s*=\s*\d{4}\b/i, "")
                   .replace(/\byear\s*=\s*\d{4}\s+AND\b/i, "")
                   .replace(/\bWHERE\s+year\s*=\s*\d{4}\b/i, "WHERE 1=1")
                   .trim();
    }
  }

  // 개별 사례 조회: ORDER BY 자동 추가 (피해 규모 순)
  if (!hasAgg && isCaseTbl && !/\bORDER\s+BY\b/i.test(clean)) {
    const orderClause = "ORDER BY damage_amount DESC, death_count DESC";
    if (/\bLIMIT\b/i.test(clean)) {
      return { sql: clean.replace(/\bLIMIT\b/i, `${orderClause} LIMIT`), note: nationalNote, isLimited: true };
    }
    return { sql: clean + ` ${orderClause} LIMIT 3`, note: nationalNote, isLimited: true };
  }

  // 집계 없는 쿼리에 LIMIT 미지정 시 자동 추가
  if (!hasAgg && !/\bLIMIT\b/i.test(clean)) return { sql: clean + " LIMIT 3", note: nationalNote, isLimited: true };
  return { sql: clean, note: nationalNote, isLimited: false };
}

// =========================
// SQL 실행
// =========================

export async function executeSQL(sql) {
  const [rows] = await Promise.race([
    pool.query(sql),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("SQL 타임아웃 (10s)")), 10000)
    ),
  ]);
  return Array.isArray(rows) ? rows : [];
}

// =========================
// 결과 포맷팅
// =========================

function fmtDate(d) {
  if (!d) return "";
  return typeof d === "object" ? d.toISOString().slice(0, 10) : String(d).slice(0, 10);
}

// 천원 단위 → 억/만/천원 자연어 변환
function fmtMoney(thousandWon) {
  if (!thousandWon) return null;
  const won = Number(thousandWon) * 1000;
  const eok = Math.floor(won / 100_000_000);
  const man = Math.floor((won % 100_000_000) / 10_000);
  if (eok > 0 && man > 0) return `${eok}억 ${man.toLocaleString()}만원`;
  if (eok > 0) return `${eok}억원`;
  if (man > 0) return `${man.toLocaleString()}만원`;
  return `${Number(thousandWon).toLocaleString()}천원`;
}

export function formatSQLResults(rows, intent) {
  if (!rows || rows.length === 0) return null;
  return intent === "stats" ? formatStats(rows) : formatCases(rows);
}

function isAggregateResult(keys) {
  // 한국어 키가 있으면 SUM/GROUP BY 별칭 결과로 판단
  if (keys.some((k) => /[ㄱ-힣]/.test(k))) return true;
  return (
    keys.some((k) => /^(count|total|sum|cnt|avg)/i.test(k)) &&
    !keys.includes("id") &&
    !keys.includes("fire_date") &&
    !keys.includes("year")
  );
}

function formatStats(rows) {
  const keys = Object.keys(rows[0]);

  if (isAggregateResult(keys)) {
    const lines = ["[집계 결과]"];
    for (const row of rows) {
      const formatted = Object.entries(row).map(([k, v]) => {
        if ((/damage|피해액|재산피해/i.test(k)) && v) return `${k}: ${fmtMoney(v)}`;
        return `${k}: ${v}`;
      });
      lines.push("- " + formatted.join(" | "));
    }
    return lines.join("\n");
  }

  // 개별 행 → stat_type별 그룹핑
  const byType = new Map();
  for (const row of rows) {
    const t = row.stat_type || "통계";
    if (!byType.has(t)) byType.set(t, []);
    byType.get(t).push(row);
  }

  const lines = [];
  for (const [type, typeRows] of byType) {
    lines.push(`\n[${type}]`);
    for (const r of typeRows) {
      const parts = [];
      if (r.fire_date) parts.push(`날짜: ${fmtDate(r.fire_date)}`);
      if (r.region && r.region !== "전국") parts.push(`지역: ${r.region}`);
      const cat = [r.category_main, r.category_sub].filter(Boolean).join(" > ");
      if (cat) parts.push(`분류: ${cat}`);
      if (r.fire_count != null) parts.push(`화재: ${r.fire_count}건`);
      if (r.death_count)  parts.push(`사망: ${r.death_count}명`);
      if (r.injury_count) parts.push(`부상: ${r.injury_count}명`);
      if (r.damage_amount) parts.push(`재산피해: ${fmtMoney(r.damage_amount)}`);
      lines.push("- " + parts.join(", "));
    }
  }
  return lines.join("\n").trim();
}

function formatCases(rows) {
  const keys = Object.keys(rows[0]);

  if (isAggregateResult(keys)) {
    const lines = ["[집계 결과]"];
    for (const row of rows) {
      const formatted = Object.entries(row).map(([k, v]) => {
        if ((/damage|피해액|재산피해/i.test(k)) && v) return `${k}: ${fmtMoney(v)}`;
        return `${k}: ${v}`;
      });
      lines.push("- " + formatted.join(" | "));
    }
    return lines.join("\n");
  }

  const lines = [`[개별 사례 ${rows.length}건]`];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    lines.push(`\n[사례 ${i + 1}]`);
    if (r.year) {
      const timeStr = r.hour != null ? ` ${r.hour}시` : "";
      lines.push(`- 발생: ${r.year}년 ${r.month}월 ${r.day}일${timeStr}`);
    }
    if (r.region) lines.push(`- 지역: ${r.region}${r.district ? " " + r.district : ""}`);
    if (r.fire_type) lines.push(`- 화재유형: ${r.fire_type}`);
    if (r.building_type && r.building_type !== "미상") lines.push(`- 건물구조: ${r.building_type}`);
    const loc = [r.location_main, r.location_mid, r.location_sub]
      .filter((v) => v && v !== "미상").join(" > ");
    if (loc) lines.push(`- 발화장소: ${loc}`);
    if (r.cause_main && r.cause_main !== "미상") {
      const cause = [r.cause_main, r.cause_sub].filter((v) => v && v !== "미상").join(" > ");
      lines.push(`- 발화요인: ${cause}`);
    }
    if (r.heat_source && r.heat_source !== "미상") lines.push(`- 발화열원: ${r.heat_source}`);
    if (r.first_object_main && r.first_object_main !== "미상") {
      const fo = [r.first_object_main, r.first_object_sub].filter((v) => v && v !== "미상").join(" > ");
      lines.push(`- 최초착화물: ${fo}`);
    }
    if (r.death_count > 0 || r.injury_count > 0) {
      lines.push(`- 인명피해: 사망 ${r.death_count}명, 부상 ${r.injury_count}명`);
    }
    if (r.damage_amount > 0) lines.push(`- 재산피해: ${fmtMoney(r.damage_amount)}`);
    if (r.suppression_time > 0) lines.push(`- 진압시간: ${Math.round(r.suppression_time / 60)}분`);
  }
  return lines.join("\n");
}

// =========================
// 메인 export
// =========================

const NATIONAL_ONLY_TYPES = new Set(["건물구조","발화열원","발화요인","발화장소","발화지점",
  "선박항공기","임야","차량발화지점","최초착화물","화재장소(전국)"]);

export async function queryBySQL(question, intent, entities) {
  const { sql, note: validateNote, isLimited } = await generateSQL(question, intent, entities);
  console.log(`[sql] ${intent}: ${sql}`);
  const rows = await executeSQL(sql);
  const rawContext = formatSQLResults(rows, intent);
  if (!rawContext) return { sql, rowCount: rows.length, context: null };

  // 전국 통계 유형인데 지역 질문이 들어온 경우 안내 메모 자동 추가
  const stMatch = sql.match(/stat_type\s*=\s*'([^']+)'/i);
  const isNationalType = stMatch && NATIONAL_ONLY_TYPES.has(stMatch[1]);
  const autoNote = (isNationalType && entities.region)
    ? `[안내: '${stMatch[1]}' 통계는 지역별 분류가 없어 전국 기준으로 제공됩니다]\n`
    : validateNote;

  // LIMIT 20 자동 적용된 경우: 전체 건수 COUNT 조회
  let totalCountNote = "";
  if (isLimited && rows.length > 0) {
    try {
      const countSql = sql
        .replace(/\bORDER\s+BY\b.*/i, "")
        .replace(/\bLIMIT\s+\d+/i, "")
        .trim();
      const countRows = await executeSQL(`SELECT COUNT(*) as total FROM (${countSql}) t`);
      const total = countRows[0]?.total ?? rows.length;
      if (total > rows.length) {
        totalCountNote = `[안내: 전체 ${total}건 중 피해 규모 기준 상위 ${rows.length}건을 표시합니다]\n`;
      }
    } catch (_) { /* COUNT 실패 시 무시 */ }
  }

  // 조회 조건 헤더 추가 (LLM이 어떤 데이터인지 인식하도록)
  const condParts = [];
  if (entities.year)   condParts.push(`${entities.year}년`);
  const loc = [entities.region, entities.district].filter(Boolean).join(" ");
  if (loc) condParts.push(loc);
  if (entities.date_ref && intent === "stats") {
    condParts.push(`날짜: ${entities.date_ref}`);
  }
  const condHeader = condParts.length > 0 ? `[조회 조건: ${condParts.join(" ")}]\n` : "";

  return { sql, rowCount: rows.length, context: condHeader + autoNote + totalCountNote + rawContext };
}
