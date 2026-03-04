/**
 * db/chat.js
 * AI 채팅 히스토리 관련 DB 모듈 (vfs DB)
 *
 * - BOT은 sender_id 문자열로만 구분 (wnsm_member 등록 불필요)
 * - room_id로 방별 대화 분리
 * - msg_type='AI'     → AI 대화 메시지 (오너 전용)
 * - msg_type='SUMMARY' → 압축 요약 (오래된 턴 대체)
 */
import { pool } from "./mysql.js";

export const BOT_MEM_ID = "bot_vfims_ai";
const HISTORY_LIMIT = 5;      // LLM에 전달할 최근 N턴
const COMPRESS_THRESHOLD = 10; // 이 턴 수 초과 시 압축 트리거

// =========================
// 오너 여부 확인
// =========================
export async function isOwner(roomId, memId) {
  const [rows] = await pool.query(
    "SELECT owner_id FROM vfs_room WHERE id = ?",
    [roomId]
  );
  if (!rows.length) return false;
  return rows[0].owner_id === memId;
}

// =========================
// AI 대화 메시지 저장 (사용자 질문 + 봇 답변)
// msg_type = 'AI' → 오너만 조회 가능
// =========================
export async function saveAiMessages(roomId, userMemId, userText, botText) {
  await pool.query(
    `INSERT INTO vfs_chat_message (room_id, sender_id, message, msg_type, send_dt)
     VALUES (?, ?, ?, 'AI', NOW()),
            (?, ?, ?, 'AI', NOW())`,
    [roomId, userMemId, userText,
     roomId, BOT_MEM_ID, botText]
  );
}

// =========================
// AI 대화 히스토리 조회 (요약 포함)
// SUMMARY 행 + 최근 HISTORY_LIMIT턴 반환 → [{user, assistant}, ...]
// =========================
export async function getAiHistory(roomId) {
  // 1. 최신 요약 조회
  const [sumRows] = await pool.query(
    `SELECT message FROM vfs_chat_message
     WHERE room_id = ? AND msg_type = 'SUMMARY' AND sender_id = ?
     ORDER BY id DESC LIMIT 1`,
    [roomId, BOT_MEM_ID]
  );
  const summary = sumRows[0]?.message || null;

  // 2. 최근 HISTORY_LIMIT 턴 조회
  const [rows] = await pool.query(
    `SELECT sender_id, message
     FROM vfs_chat_message
     WHERE room_id = ? AND msg_type = 'AI'
     ORDER BY id DESC LIMIT ?`,
    [roomId, HISTORY_LIMIT * 2]
  );
  rows.reverse();

  const history = [];
  let turn = {};
  for (const row of rows) {
    if (row.sender_id !== BOT_MEM_ID) {
      turn = { user: row.message };
    } else if (turn.user !== undefined) {
      history.push({ user: turn.user, assistant: row.message });
      turn = {};
    }
  }

  // 3. 요약이 있으면 첫 번째 턴으로 삽입
  if (summary) {
    history.unshift({ user: "[이전 대화 요약]", assistant: summary });
  }

  return history;
}

// =========================
// 화면 표시용 최근 메시지 조회 (페이지 로드 시)
// =========================
export async function getRecentAiMessages(roomId, limit = 20) {
  const [rows] = await pool.query(
    `SELECT sender_id, message FROM vfs_chat_message
     WHERE room_id = ? AND msg_type = 'AI'
     ORDER BY id DESC LIMIT ?`,
    [roomId, limit * 2]
  );
  rows.reverse();

  const turns = [];
  let turn = {};
  for (const row of rows) {
    if (row.sender_id !== BOT_MEM_ID) {
      turn = { user: row.message };
    } else if (turn.user !== undefined) {
      turns.push({ user: turn.user, assistant: row.message });
      turn = {};
    }
  }
  return turns;
}

// =========================
// 압축 필요 여부 확인
// =========================
export async function countAiTurns(roomId) {
  const [rows] = await pool.query(
    `SELECT COUNT(*) AS cnt FROM vfs_chat_message
     WHERE room_id = ? AND msg_type = 'AI' AND sender_id != ?`,
    [roomId, BOT_MEM_ID]
  );
  return Number(rows[0]?.cnt || 0);
}

export { COMPRESS_THRESHOLD };

// =========================
// 압축 대상 턴 조회 (최근 HISTORY_LIMIT 제외한 오래된 것들)
// =========================
export async function getHistoryForCompression(roomId) {
  const [rows] = await pool.query(
    `SELECT id, sender_id, message FROM vfs_chat_message
     WHERE room_id = ? AND msg_type = 'AI'
     ORDER BY id ASC`,  // ARCHIVED는 이미 제외됨
    [roomId]
  );

  const keepCount = HISTORY_LIMIT * 2;
  if (rows.length <= keepCount) return { turns: [], ids: [] };

  const toCompress = rows.slice(0, rows.length - keepCount);
  const ids = toCompress.map((r) => r.id);

  const turns = [];
  let turn = {};
  for (const row of toCompress) {
    if (row.sender_id !== BOT_MEM_ID) {
      turn = { user: row.message };
    } else if (turn.user !== undefined) {
      turns.push({ user: turn.user, assistant: row.message });
      turn = {};
    }
  }

  return { turns, ids };
}

// =========================
// 요약 저장 + 압축된 AI 행 삭제 (트랜잭션)
// =========================
export async function saveSummaryAndPrune(roomId, summaryText, idsToDelete) {
  const conn = await pool.getConnection();
  try {
    await conn.beginTransaction();
    // 기존 요약 교체
    await conn.query(
      `DELETE FROM vfs_chat_message WHERE room_id = ? AND msg_type = 'SUMMARY'`,
      [roomId]
    );
    await conn.query(
      `INSERT INTO vfs_chat_message (room_id, sender_id, message, msg_type)
       VALUES (?, ?, ?, 'SUMMARY')`,
      [roomId, BOT_MEM_ID, summaryText]
    );
    // 압축된 AI 행 → ARCHIVED로 보존 (삭제 안 함)
    if (idsToDelete.length > 0) {
      await conn.query(
        `UPDATE vfs_chat_message SET msg_type = 'ARCHIVED' WHERE id IN (?)`,
        [idsToDelete]
      );
    }
    await conn.commit();
  } catch (e) {
    await conn.rollback();
    throw e;
  } finally {
    conn.release();
  }
}
