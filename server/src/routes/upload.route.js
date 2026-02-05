import { Router } from "express";
import multer from "multer";
import { handleUpload } from "../handlers/upload.handler.js";

const router = Router();

// Upload configuration (memory storage, 100MB limit)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 },
});

/**
 * POST /upload
 * File upload endpoint
 */
router.post("/upload", upload.single("file"), handleUpload);

export default router;
