/**
 * vfs_data_api → 월별 × stat_type 단위 문서 → bge-m3 임베딩 → ChromaDB 적재
 *
 * 문서 구조: 연월 × stat_type = 1개 문서 (해당 월 전체 카테고리 포함)
 * 예) "2024년 6월 발화요인별 화재 통계: - 부주의(담배꽁초): 684건 ..."
 *
 * 실행: node scripts/ingest_api_stats.js
 * 옵션:
 *   --from=2020   특정 연도부터만 적재
 *   --dry-run     임베딩/적재 없이 문장 변환 결과만 확인
 */

import "dotenv/config";
import mysql from "mysql2/promise";
import { ChromaClient } from "chromadb";
import crypto from "node:crypto";

// ---- 설정 ----
const MYSQL = {
  host: process.env.MY_HOST || "218.158.62.155",
  port: Number(process.env.MY_PORT) || 33307,
  user: process.env.MY_USER || "vfs",
  password: process.env.MY_PASS,
  database: process.env.MY_DB || "vfs",
  charset: "utf8mb4",
};

const EMB_URL = (process.env.EMB_URL || "http://localhost:8001").replace(/\/$/, "");
const EMB_MODEL = process.env.EMB_MODEL || "BAAI/bge-m3";

const _chromaUrl = new URL(process.env.CHROMA_URL || "http://localhost:8003");
const CHROMA = {
  host: _chromaUrl.hostname,
  port: Number(_chromaUrl.port) || (_chromaUrl.protocol === "https:" ? 443 : 80),
  ssl: _chromaUrl.protocol === "https:",
};

const COLLECTION = "vfims_stats";
const BATCH_SIZE = 32;
const MAX_CATEGORIES = 30; // 카테고리 수 많은 stat_type(발화지점 252개 등) 상위 N개만

// CLI 옵션
const args = Object.fromEntries(
  process.argv.slice(2)
    .filter((a) => a.startsWith("--"))
    .map((a) => a.slice(2).split("="))
    .map(([k, v]) => [k, v ?? true])
);
const FROM_YEAR = args.from ? Number(args.from) : null;
const TARGET_DATE = args.date || null; // "YYYY-MM-DD" 형식, 특정 날만 처리
const DRY_RUN = !!args["dry-run"];

// ================================================================
// stat_type별 헤더 문구
// ================================================================
const STAT_HEADERS = {
  발화요인:    "발화요인별",
  최초착화물:  "최초착화물별",
  발화열원:    "발화열원별",
  발화장소:    "발화장소별",
  발화지점:    "발화지점별",
  건물구조:    "건물구조별",
  차량발화지점: "차량 발화지점별",
  임야:        "임야 화재유형별",
  선박항공기:  "선박·항공기 화재유형별",
};

// ================================================================
// 포맷 헬퍼
// ================================================================
function fmt(n) {
  return Number(n).toLocaleString("ko-KR");
}

function fmtDmg(v) {
  v = Number(v) * 1000; // DB 저장 단위: 천원 → 원으로 변환
  if (!v) return "없음";
  if (v >= 100_000_000) return `${(v / 100_000_000).toFixed(1)}억원`;
  if (v >= 10_000)      return `${Math.round(v / 10_000).toLocaleString("ko-KR")}만원`;
  return `${v.toLocaleString("ko-KR")}원`;
}

// ================================================================
// 문서 생성: (year, month, stat_type, categories[]) → 자연어 문자열
// ================================================================
function buildDocument(year, month, day, stat_type, categories) {
  const date   = day ? `${year}년 ${month}월 ${day}일` : `${year}년 ${month}월`;
  const header = STAT_HEADERS[stat_type] ?? `${stat_type}별`;

  // 합계
  const totalFire   = categories.reduce((s, r) => s + Number(r.fire_count),    0);
  const totalDeath  = categories.reduce((s, r) => s + Number(r.death_count),   0);
  const totalInjury = categories.reduce((s, r) => s + Number(r.injury_count),  0);
  const totalDamage = categories.reduce((s, r) => s + Number(r.damage_amount), 0);

  // 상위 N개 카테고리 (이미 fire_count DESC 정렬)
  const shown = categories.slice(0, MAX_CATEGORIES);
  const rest  = categories.length - shown.length;

  let lines = [`${date} ${header} 화재 통계:`];

  for (const r of shown) {
    let line = `- ${r.category}: ${fmt(r.fire_count)}건`;
    if (Number(r.death_count)  > 0) line += `, 사망 ${r.death_count}명`;
    if (Number(r.injury_count) > 0) line += `, 부상 ${r.injury_count}명`;
    line += `, 재산피해 ${fmtDmg(r.damage_amount)}`;
    lines.push(line);
  }

  if (rest > 0) {
    lines.push(`- 외 ${rest}개 항목 포함`);
  }

  lines.push(
    `전체 합계: ${fmt(totalFire)}건, 사망 ${totalDeath}명, 부상 ${totalInjury}명, 재산피해 ${fmtDmg(totalDamage)}.`
  );

  return lines.join("\n");
}

// ================================================================
// 문서 ID
// ================================================================
function makeId(year, month, day, stat_type) {
  const key = `${year}_${String(month).padStart(2, "0")}_${String(day).padStart(2, "0")}_${stat_type}`;
  const hash = crypto.createHash("md5").update(key).digest("hex").slice(0, 6);
  return `stats_${year}_${String(month).padStart(2, "0")}_${String(day).padStart(2, "0")}_${hash}`;
}

// ================================================================
// 임베딩
// ================================================================
async function embedBatch(texts) {
  const res = await fetch(`${EMB_URL}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMB_MODEL, input: texts }),
  });
  if (!res.ok) throw new Error(`Embedding 오류 ${res.status}: ${await res.text()}`);
  const json = await res.json();
  return json.data.map((d) => d.embedding);
}

// ================================================================
// 메인
// ================================================================
async function main() {
  console.log("=".repeat(60));
  console.log("📥  vfs_data_api → ChromaDB 적재 파이프라인");
  console.log("    문서 단위: 연월 × stat_type (카테고리 전체 포함)");
  console.log("=".repeat(60));
  if (DRY_RUN)     console.log("🔍  DRY-RUN 모드");
  if (TARGET_DATE) console.log(`📅  특정 날짜만 처리: ${TARGET_DATE}`);
  else if (FROM_YEAR) console.log(`📅  ${FROM_YEAR}년 이후만 처리`);
  console.log();

  // MySQL 연결
  const db = await mysql.createConnection(MYSQL);
  console.log("✅  MySQL 연결:", MYSQL.host);

  // ChromaDB 연결
  let col;
  if (!DRY_RUN) {
    const chroma = new ChromaClient(CHROMA);
    col = await chroma.getOrCreateCollection({
      name: COLLECTION,
      embeddingFunction: null,
      metadata: { "hnsw:space": "cosine" },
    });
    console.log("✅  ChromaDB 컬렉션:", COLLECTION);
  }
  console.log();

  // 카테고리별 집계 (fire_count DESC 정렬)
  const where = TARGET_DATE
    ? `WHERE DATE(fire_date) = '${TARGET_DATE}'`
    : FROM_YEAR ? `WHERE YEAR(fire_date) >= ${FROM_YEAR}` : "";
  const [rows] = await db.query(`
    SELECT
      YEAR(fire_date)    AS year,
      MONTH(fire_date)   AS month,
      DAY(fire_date)     AS day,
      stat_type,
      category,
      SUM(fire_count)    AS fire_count,
      SUM(death_count)   AS death_count,
      SUM(injury_count)  AS injury_count,
      SUM(damage_amount) AS damage_amount
    FROM vfs_data_api
    ${where}
    GROUP BY YEAR(fire_date), MONTH(fire_date), DAY(fire_date), stat_type, category
    ORDER BY year, month, day, stat_type, fire_count DESC
  `);
  console.log(`📊  카테고리별 집계: ${rows.length.toLocaleString()}행`);

  // 연월 × stat_type 단위로 그룹핑
  const groups = new Map();
  for (const r of rows) {
    const key = `${r.year}_${r.month}_${r.day}_${r.stat_type}`;
    if (!groups.has(key)) groups.set(key, { year: r.year, month: r.month, day: r.day, stat_type: r.stat_type, categories: [] });
    groups.get(key).categories.push(r);
  }

  // 문서 생성
  const docs = [];
  for (const { year, month, day, stat_type, categories } of groups.values()) {
    const text = buildDocument(year, month, day, stat_type, categories);
    const totalFire   = categories.reduce((s, r) => s + Number(r.fire_count),    0);
    const totalDeath  = categories.reduce((s, r) => s + Number(r.death_count),   0);
    const totalInjury = categories.reduce((s, r) => s + Number(r.injury_count),  0);
    const totalDamage = categories.reduce((s, r) => s + Number(r.damage_amount), 0);

    docs.push({
      id:   makeId(year, month, day, stat_type),
      text,
      meta: {
        doc_type:      "stats",
        stat_type,
        year:          Number(year),
        month:         Number(month),
        day:           Number(day),
        category_count: categories.length,
        fire_count:    totalFire,
        death_count:   totalDeath,
        injury_count:  totalInjury,
        damage_amount: totalDamage,
      },
    });
  }

  console.log(`📄  생성 문서 수: ${docs.length.toLocaleString()}개`);
  console.log(`    (카테고리 ${rows.length.toLocaleString()}행 → 문서 ${docs.length.toLocaleString()}개로 압축)\n`);

  // DRY-RUN: 샘플 출력
  if (DRY_RUN) {
    const samples = docs.filter((_, i) => i % Math.floor(docs.length / 9) === 0).slice(0, 9);
    samples.forEach((d, i) => {
      console.log(`\n[${ i + 1}] ID: ${d.id}  |  stat_type: ${d.meta.stat_type}`);
      console.log("─".repeat(60));
      console.log(d.text);
    });
    await db.end();
    return;
  }

  // 배치 임베딩 + ChromaDB upsert
  console.log(`⏳  임베딩 + 적재 시작 (배치: ${BATCH_SIZE})\n`);
  let done = 0;
  const t0 = Date.now();

  for (let i = 0; i < docs.length; i += BATCH_SIZE) {
    const batch = docs.slice(i, i + BATCH_SIZE);
    const embeddings = await embedBatch(batch.map((d) => d.text));

    await col.upsert({
      ids:       batch.map((d) => d.id),
      documents: batch.map((d) => d.text),
      embeddings,
      metadatas: batch.map((d) => d.meta),
    });

    done += batch.length;
    const pct     = Math.round((done / docs.length) * 100);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
    const eta     = done < docs.length
      ? Math.round(((Date.now() - t0) / done) * (docs.length - done) / 1000)
      : 0;
    process.stdout.write(
      `\r  진행: ${done}/${docs.length} (${pct}%) | 경과: ${elapsed}s | 남은 예상: ${eta}s   `
    );
  }

  console.log(`\n\n✅  적재 완료!`);
  console.log(`   - 총 문서:   ${docs.length.toLocaleString()}개`);
  console.log(`   - 소요 시간: ${((Date.now() - t0) / 1000).toFixed(1)}초`);
  console.log(`   - 컬렉션:    ${COLLECTION}`);

  await db.end();
}

main().catch((e) => {
  console.error("\n❌  오류:", e.message);
  process.exit(1);
});
