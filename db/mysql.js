const mysql = require("mysql2/promise");
const dotenv = require("dotenv");
const path = require("path");

dotenv.config({ path: path.join(__dirname, "..", ".env") });

const pool = mysql.createPool({
  host: process.env.MY_HOST,
  port: Number(process.env.MY_PORT),
  user: process.env.MY_USER,
  password: process.env.MY_PASS,
  database: process.env.MY_DB,
  charset: "utf8mb4",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  timezone: "Z",
});

async function getConn() {
  return pool.getConnection();
}

module.exports = { pool, getConn };
