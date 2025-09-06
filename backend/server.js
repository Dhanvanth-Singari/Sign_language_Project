import express from "express";
import multer from "multer";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import ffmpeg from "fluent-ffmpeg";
import { Client } from "@gradio/client";

ffmpeg.setFfmpegPath("C:\\Users\\91949\\Downloads\\ffmpeg-8.0-essentials_build\\ffmpeg-8.0-essentials_build\\bin\\ffmpeg.exe");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());

const uploadPath = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadPath)) fs.mkdirSync(uploadPath);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadPath),
  filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname),
});
const upload = multer({ storage });

app.post("/upload", upload.single("video"), async (req, res) => {
  let webmPath, mp4Path;
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    webmPath = req.file.path;
    mp4Path = path.join(uploadPath, `video_${Date.now()}.mp4`);

    console.log("🎥 Converting WebM to MP4...");
    console.log("📂 Input:", webmPath);
    console.log("📂 Output:", mp4Path);

    // Convert WebM to MP4
    await new Promise((resolve, reject) => {
      ffmpeg(webmPath)
        .videoCodec("libx264")
        .outputOptions(["-movflags +faststart", "-y"])
        .toFormat("mp4")
        .on("end", () => {
          console.log("✅ Video conversion completed");
          resolve();
        })
        .on("error", (err) => {
          console.error("❌ FFmpeg error:", err);
          reject(err);
        })
        .save(mp4Path);
    });

    // Verify file was created
    if (!fs.existsSync(mp4Path)) {
      throw new Error("MP4 file was not created");
    }

    console.log("📊 File size:", fs.statSync(mp4Path).size, "bytes");
    console.log("🌐 Connecting to Hugging Face API...");

    // Connect to your friend's Gradio space
    const client = await Client.connect("Anudeep05/VideoMAE-API");
    console.log("✅ Connected successfully");

    // IMPORTANT: This is the correct way to call your friend's API
    // Looking at app.py, the function expects a File object
    const result = await client.predict("/predict", {
      video_file: {
        path: mp4Path,
        meta: {"_type": "gradio.FileData"}
      }
    });

    console.log("🎯 Raw API result:", result);

    let prediction = "No prediction available";
    
    // Extract prediction from result
    if (result && result.data) {
      if (Array.isArray(result.data)) {
        prediction = result.data[0] || "No prediction";
      } else {
        prediction = result.data || "No prediction";
      }
    }

    console.log("🎯 Final prediction:", prediction);

    // Clean up files
    if (fs.existsSync(webmPath)) {
      fs.unlinkSync(webmPath);
      console.log("🧹 Cleaned WebM file");
    }
    if (fs.existsSync(mp4Path)) {
      fs.unlinkSync(mp4Path);
      console.log("🧹 Cleaned MP4 file");
    }

    res.json({ 
      prediction: prediction,
      success: true,
      message: "Video processed successfully"
    });

  } catch (err) {
    console.error("💥 SERVER ERROR:", err);
    
    // Clean up on error
    try {
      if (webmPath && fs.existsSync(webmPath)) fs.unlinkSync(webmPath);
      if (mp4Path && fs.existsSync(mp4Path)) fs.unlinkSync(mp4Path);
    } catch (cleanupError) {
      console.error("🧹 Cleanup error:", cleanupError);
    }
    
    res.status(500).json({ 
      error: "Processing failed", 
      details: err.message || "Unknown error"
    });
  }
});

// Test endpoint
app.get("/test", (req, res) => {
  res.json({ 
    status: "Server working ✅", 
    time: new Date().toISOString()
  });
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
  console.log(`📁 Upload directory: ${uploadPath}`);
});