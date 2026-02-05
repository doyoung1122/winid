FROM node:20

WORKDIR /app

# 1. 의존성 파일 복사
COPY server/package.json ./server/
COPY db/ ./db/

# 2. 의존성 설치
WORKDIR /app/server
RUN npm install

# 3. 소스 코드 전체 복사
WORKDIR /app
COPY . .

# 4. 포트 노출 (메인 서버)
EXPOSE 8000

# 5. 실행 명령어
WORKDIR /app/server
CMD ["node", "src/index.js"]
