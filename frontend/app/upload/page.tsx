"use client"

import { Navbar } from "@/components/navbar"
import { UploadArea } from "@/components/upload-area"
import { CodeOutput } from "@/components/code-output"
import { useState } from "react"

export default function UploadPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "/_/backend"

  const [generatedCode, setGeneratedCode] = useState<{
    html?: string
    css?: string
    react?: string
    raw?: string
  } | null>(null)

  const [isGenerating, setIsGenerating] = useState(false)

  const handleImageUpload = async (file: File) => {
    setIsGenerating(true)

    try {
      // Create form data to send to backend
      const formData = new FormData()
      formData.append("image", file) // ✔ BACKEND EXPECTS "image", not "file"

      // Send request to backend
      const res = await fetch(`${apiBase}/api/generate`, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`Backend error: ${await res.text()}`)
      }

      const data = await res.json() // ✔ FIXED: correct variable
      console.log("Frontend received:", data);
      setGeneratedCode(data)
    } catch (error) {
      console.error("Failed to generate code:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="min-h-screen w-full dark bg-background">
      <Navbar />

      <div className="pt-20 pb-12">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-foreground mb-2">
              Generate Code from Design
            </h1>
            <p className="text-muted-foreground">
              Upload a design image to get started
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Upload Section */}
            <div className="flex flex-col">
              <UploadArea
                onUpload={handleImageUpload}
                isLoading={isGenerating}
              />
            </div>

            {/* Code Output Section */}
            <div className="flex flex-col">
              {generatedCode ? (
                <CodeOutput code={generatedCode} />
              ) : (
                <div className="p-8 rounded-lg border border-border/40 bg-card flex items-center justify-center h-full min-h-[500px]">
                  <div className="text-center">
                    <div className="text-4xl mb-4 text-muted-foreground">📝</div>
                    <p className="text-muted-foreground">
                      Generated code will appear here
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
