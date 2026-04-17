process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";
import express from "express";
import multer from "multer";
import fs from "fs/promises";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

const app = express();
app.use(cors());
const upload = multer({ dest: "uploads/" });

// POST /api/generate
app.post("/api/generate", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image uploaded" });
  }

  try {
    console.log("Processing upload:", req.file.path);
    const buffer = await fs.readFile(req.file.path);
    const base64 = buffer.toString("base64");

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [
        {
          parts: [
            {
              inlineData: {
                data: base64,
                mimeType: "image/png",
              },
            },
            {
              text: `
You are an expert frontend engineer. Generate clean HTML, CSS, and React code from this UI screenshot.

Output as pure JSON:
{
  "html": "...",
  "css": "...",
  "react": "..."
}
              `,
            },
          ],
        },
      ],
    });

    // Extract the generated text properly
    const text =
      response?.candidates?.[0]?.content?.parts?.[0]?.text || "";

    console.log("Gemini Raw Response:", text); // LOG RAW RESPONSE

    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      console.error("JSON Parse Error:", e);
      // Fallback: Try to clean markdown blocks if present
      const cleanText = text.replace(/```json/g, "").replace(/```/g, "").trim();
      try {
        parsed = JSON.parse(cleanText);
      } catch (e2) {
        console.error("Secondary JSON Parse Error:", e2);
        console.log("Attempting Regex Extraction...");

        const extract = (key) => {
          // Robust regex to handle even raw newlines inside the JSON string
          const regex = new RegExp(`"${key}"\\s*:\\s*"([\\s\\S]*?)"(?=\\s*,|\\s*})`, 'g');
          const match = regex.exec(cleanText);
          return match ? match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"') : "";
        };

        const extractMarkdown = (lang) => {
          const regex = new RegExp(`\`\`\`(?:${lang})\\s*([\\s\\S]*?)\`\`\``, 'i');
          const match = text.match(regex);
          return match ? match[1].trim() : "";
        }

        let html = extract("html");
        let css = extract("css");
        let react = extract("react");

        console.log("Regex JSON Extract Results - HTML:", !!html, "CSS:", !!css, "React:", !!react);

        if (!html) html = extractMarkdown("html");
        if (!css) css = extractMarkdown("css");
        if (!react) react = extractMarkdown("jsx|javascript|js|react|tsx");

        console.log("Final Extract Results - HTML:", !!html, "CSS:", !!css, "React:", !!react);

        if (html || css || react) {
          parsed = { html, css, react, raw: text };
        } else {
          parsed = { raw: text };
        }
      }
    }

    await fs.unlink(req.file.path);

    // Final check: if all code fields are empty, send raw anyway
    if (parsed && !parsed.html && !parsed.css && !parsed.react) {
      parsed.raw = text;
    }

    res.json(parsed);

  } catch (error) {
    console.error("Backend error:", error);
    res.status(500).json({ error: error.message });
  }
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Backend running at http://localhost:${PORT}`);
});
