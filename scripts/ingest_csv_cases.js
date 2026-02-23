/**
 * vfs_data_csv → 개별 화재 사례 문서 → bge-m3 임베딩 → ChromaDB 적재
 *
 * 문서 구조: 개별 화재 사례 1건 = 1문서
 * 예) "2022년 9월 15일 오후 3시 경상북도 문경시 임야(과수원)에서
 *      부주의(쓰레기 소각)로 화재 발생. 발화열원: 불꽃·불티.
 *      최초착화물: 종이·목재(풀, 나뭇잎). 진압 34분, 재산피해 없음."
 *
 * 실행: node scripts/ingest_csv_cases.js
 * 옵션:
 *   --from=2022   특정 연도부터만 적재
 *   --dry-run     임베딩/적재 없이 문장 변환 결과만 확인 (샘플 9건)
 */

import "dotenv/config";
import mysql from "mysql2/promise";
import { ChromaClient } from "chromadb";
import crypto from "node:crypto";

// ================================================================
// 설정
// ================================================================
const MYSQL = {
  host: process.env.MY_HOST || "218.158.62.155",
  port: Number(process.env.MY_PORT) || 33307,
  user: process.env.MY_USER || "vfs",
  password: process.env.MY_PASS,
  database: process.env.MY_DB || "vfs",
  charset: "utf8mb4",
};

const EMB_URL   = (process.env.EMB_URL  || "http://localhost:8001").replace(/\/$/, "");
const EMB_MODEL = process.env.EMB_MODEL || "BAAI/bge-m3";

const _chromaUrl = new URL(process.env.CHROMA_URL || "http://localhost:8003");
const CHROMA = {
  host: _chromaUrl.hostname,
  port: Number(_chromaUrl.port) || (_chromaUrl.protocol === "https:" ? 443 : 80),
  ssl:  _chromaUrl.protocol === "https:",
};

const COLLECTION = "vfims_documents";
const BATCH_SIZE = 64;

// CLI 옵션
const args = Object.fromEntries(
  process.argv.slice(2)
    .filter((a) => a.startsWith("--"))
    .map((a) => a.slice(2).split("="))
    .map(([k, v]) => [k, v ?? true])
);
const FROM_YEAR = args.from ? Number(args.from) : null;
const DRY_RUN   = !!args["dry-run"];

// ================================================================
// 포맷 헬퍼
// ================================================================
const isMissing = (v) => !v || v === "미상" || v === "미상(미상)";

function fmtDate(year, month, day, hour) {
  let s = `${year}년 ${month}월 ${day}일`;
  if (hour != null) {
    const ampm = hour < 12 ? "오전" : "오후";
    const h    = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour;
    s += ` ${ampm} ${h}시`;
  }
  return s;
}

function fmtTime(seconds) {
  if (!seconds || seconds <= 0) return null;
  const min = Math.round(seconds / 60);
  return min < 60 ? `${min}분` : `${Math.floor(min / 60)}시간 ${min % 60}분`;
}

function fmtDmg(v) {
  v = Number(v);
  if (!v) return null;
  if (v >= 100_000_000) return `${(v / 100_000_000).toFixed(1)}억원`;
  if (v >= 10_000)      return `${Math.round(v / 10_000).toLocaleString("ko-KR")}만원`;
  return `${v.toLocaleString("ko-KR")}원`;
}

function fmtLocation(main, mid, sub) {
  const parts = [main, mid, sub].filter((v) => !isMissing(v));
  // 중복 제거 (mid === sub 인 경우 있음)
  const unique = [...new Set(parts)];
  return unique.join(" > ");
}

function fmtCause(main, sub) {
  if (isMissing(sub) || sub === main) return main;
  return `${main}(${sub})`;
}

function fmtTempWind(temp, wind, humidity) {
  const parts = [];
  if (temp != null)     parts.push(`기온 ${temp}도`);
  if (!isMissing(wind)) parts.push(`풍속 ${wind}`);
  if (humidity != null) parts.push(`습도 ${humidity}%`);
  return parts.join(", ");
}

// ================================================================
// 문서 생성: 1행 → 자연어 문장
// ================================================================
function buildDocument(r) {
  const lines = [];

  // 1. 날짜 + 지역 + 발화장소 + 발화원인
  const date     = fmtDate(r.year, r.month, r.day, r.hour);
  const location = fmtLocation(r.location_main, r.location_mid, r.location_sub);
  const cause    = fmtCause(r.cause_main, r.cause_sub);

  let first = `${date} ${r.region} ${r.district}`;
  if (!isMissing(location)) first += ` ${location}`;
  first += `에서 ${cause}으로 화재가 발생했습니다.`;
  lines.push(first);

  // 2. 발화열원
  if (!isMissing(r.heat_source)) {
    lines.push(`발화열원: ${r.heat_source}.`);
  }

  // 3. 발화기구 (미상 58.9% → 있을 때만)
  if (!isMissing(r.device_main)) {
    const device = isMissing(r.device_sub) || r.device_sub === r.device_main
      ? r.device_main
      : `${r.device_main}(${r.device_sub})`;
    lines.push(`발화기구: ${device}.`);
  }

  // 4. 최초착화물
  if (!isMissing(r.first_object_main)) {
    const obj = isMissing(r.first_object_sub) || r.first_object_sub === r.first_object_main
      ? r.first_object_main
      : `${r.first_object_main}(${r.first_object_sub})`;
    lines.push(`최초착화물: ${obj}.`);
  }

  // 5. 건물 구조 (미상 35.3% → 있을 때만)
  if (!isMissing(r.building_type)) {
    lines.push(`건물구조: ${r.building_type}.`);
  }

  // 6. 진압시간 + 재산피해
  const suppParts = [];
  const suppTime  = fmtTime(r.suppression_time);
  if (suppTime) suppParts.push(`진압시간 ${suppTime}`);
  const dmg = fmtDmg(r.damage_amount);
  suppParts.push(dmg ? `재산피해 ${dmg}` : `재산피해 없음`);
  lines.push(suppParts.join(", ") + ".");

  // 7. 인명피해
  const death   = Number(r.death_count)   || 0;
  const injury  = Number(r.injury_count)  || 0;
  if (death > 0 || injury > 0) {
    const dmgParts = [];
    if (death  > 0) dmgParts.push(`사망 ${death}명`);
    if (injury > 0) dmgParts.push(`부상 ${injury}명`);
    lines.push(`인명피해: ${dmgParts.join(", ")}.`);
  }

  // 8. 기상
  const weather = fmtTempWind(r.temperature, r.wind_speed, r.humidity);
  if (weather) lines.push(`기상: ${weather}.`);

  return lines.join(" ");
}

// ================================================================
// 문서 ID
// ================================================================
function makeId(r) {
  // id(PK)로 항상 유니크 보장, report_no는 접두사로만 활용
  const prefix = r.report_no?.trim() ? r.report_no.trim() : "x";
  return `case_${prefix}_${r.id}`;
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
  console.log("📥  vfs_data_csv → ChromaDB 개별 사례 적재 파이프라인");
  console.log("    문서 단위: 개별 화재 사례 1건 = 1문서");
  console.log("=".repeat(60));
  if (DRY_RUN)   console.log("🔍  DRY-RUN 모드");
  if (FROM_YEAR) console.log(`📅  ${FROM_YEAR}년 이후만 처리`);
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

  // 데이터 로드
  const where = FROM_YEAR ? `WHERE year >= ${FROM_YEAR}` : "";
  const [rows] = await db.query(`
    SELECT id, report_no, fire_type,
           building_type,
           location_main, location_mid, location_sub,
           cause_main, cause_sub,
           heat_source,
           device_main, device_sub,
           first_object_main, first_object_sub,
           death_count, injury_count, damage_amount,
           suppression_time,
           region, district,
           temperature, wind_speed, humidity,
           year, month, day, hour
    FROM vfs_data_csv
    ${where}
    ORDER BY year, month, day, id
  `);

  console.log(`📊  로드: ${rows.length.toLocaleString()}건`);

  // 문서 생성
  const docs = rows.map((r) => ({
    id:   makeId(r),
    text: buildDocument(r),
    meta: {
      doc_type:      "case",
      type:          "case",
      report_no:     r.report_no || "",
      year:          Number(r.year),
      month:         Number(r.month),
      region:        r.region || "",
      district:      r.district || "",
      cause_main:    r.cause_main || "",
      location_main: r.location_main || "",
      fire_type:     r.fire_type || "",
      death_count:   Number(r.death_count)  || 0,
      injury_count:  Number(r.injury_count) || 0,
      damage_amount: Number(r.damage_amount) || 0,
    },
  }));

  console.log(`📄  생성 문서 수: ${docs.length.toLocaleString()}개\n`);

  // DRY-RUN: 샘플 출력
  if (DRY_RUN) {
    const step = Math.max(1, Math.floor(docs.length / 9));
    const samples = docs.filter((_, i) => i % step === 0).slice(0, 9);
    samples.forEach((d, i) => {
      console.log(`\n[${i + 1}] ID: ${d.id}`);
      console.log(`     cause: ${d.meta.cause_main} | region: ${d.meta.region} | ${d.meta.year}년 ${d.meta.month}월`);
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
    const batch      = docs.slice(i, i + BATCH_SIZE);
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
      `\r  진행: ${done.toLocaleString()}/${docs.length.toLocaleString()} (${pct}%) | 경과: ${elapsed}s | 남은 예상: ${eta}s   `
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
