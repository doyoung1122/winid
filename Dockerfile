FROM node:20

WORKDIR /app

# 1. 시스템 패키지 및 Python 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Python 의존성 설치 (Docling)
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    pandas \
    "docling[pdf]" \
    docling-core

# 3. Node.js 의존성 파일 복사
COPY package.json ./
COPY server/package.json ./server/
COPY db/ ./db/

# 4. Node.js 의존성 설치
RUN npm install
WORKDIR /app/server
RUN npm install

# 5. 소스 코드 전체 복사
WORKDIR /app
COPY . .

# 6. 포트 노출 (메인 서버)
EXPOSE 8000

# 7. 실행 명령어
WORKDIR /app/server
CMD ["node", "src/index.js"]
