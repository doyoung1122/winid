import mysql from "mysql2/promise";
import { MY_HOST, MY_PORT, MY_USER, MY_PASS, MY_DB } from "../server/src/config/env.js";

const pool = mysql.createPool({
  host: MY_HOST,
  port: MY_PORT,
  user: MY_USER,
  password: MY_PASS,
  database: MY_DB,
  charset: "utf8mb4",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  timezone: "Z",
});

async function getConn() {
  return pool.getConnection();
}

export { pool, getConn };
