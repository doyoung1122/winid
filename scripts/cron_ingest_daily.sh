#!/bin/bash
# 매일 10:30 실행 - 전날 vfs_data_api 데이터를 ChromaDB에 적재
# crontab: 30 10 * * * /home/winidsoft/vfims_llm/scripts/cron_ingest_daily.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
LOG_FILE="$LOG_DIR/ingest_${YESTERDAY}.log"

mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 적재 시작: $YESTERDAY" | tee "$LOG_FILE"

cd "$PROJECT_DIR"
node scripts/ingest_api_stats.js --date="$YESTERDAY" >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 적재 완료" | tee -a "$LOG_FILE"
