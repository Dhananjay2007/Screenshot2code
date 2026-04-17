import { GoogleGenAI } from "@google/genai"
import dotenv from "dotenv"
dotenv.config()

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
})

const res = await ai.models.list()
console.log(res)
