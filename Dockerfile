
FROM node:20

WORKDIR /app

# 1. 시스템 패키지 업데이트 및 필수 도구 설치
COPY server/package.json ./server/
COPY mcp_server/package.json ./mcp_server/

# 2. 의존성 설치
WORKDIR /app/server
RUN npm install
WORKDIR /app/mcp_server
RUN npm install

# 3. 소스 코드 전체 복사
WORKDIR /app
COPY . .

# 4. 포트 노출 (메인 서버)
EXPOSE 8000

# 5. 실행 명령어 (기본적으로 server/index.js 실행)
WORKDIR /app/server
CMD ["node", "index.js"]