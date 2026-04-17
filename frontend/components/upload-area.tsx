"use client"

import type React from "react"

import { useState, useRef } from "react"
import gsap from "gsap"

interface UploadAreaProps {
  onUpload: (file: File) => void
  isLoading?: boolean
}

export function UploadArea({ onUpload, isLoading = false }: UploadAreaProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const uploadBoxRef = useRef<HTMLDivElement>(null)
  const loadingRef = useRef<HTMLDivElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const processFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file")
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target?.result as string)
      onUpload(file)
    }
    reader.readAsDataURL(file)

    if (loadingRef.current) {
      gsap.fromTo(
        loadingRef.current,
        { opacity: 0, scale: 0.8 },
        { opacity: 1, scale: 1, duration: 0.3, ease: "back.out" },
      )

      // Spin animation for loader
      gsap.to(loadingRef.current.querySelector(".spinner"), {
        rotation: 360,
        duration: 1,
        repeat: -1,
        ease: "none",
      })
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      processFile(files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (files && files.length > 0) {
      processFile(files[0])
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  if (isLoading) {
    return (
      <div ref={loadingRef} className="p-8 rounded-lg border border-border/40 bg-card">
        <div className="flex flex-col items-center justify-center min-h-[300px]">
          <div className="spinner w-12 h-12 border-2 border-accent/30 border-t-accent rounded-full mb-4"></div>
          <p className="text-foreground font-medium">Generating code...</p>
          <p className="text-muted-foreground text-sm mt-1">This might take a moment</p>
        </div>
      </div>
    )
  }

  if (preview) {
    return (
      <div className="rounded-lg border border-border/40 bg-card overflow-hidden flex flex-col h-full">
        <div className="p-4 border-b border-border/40 flex items-center justify-between">
          <h3 className="font-semibold text-foreground">Upload Preview</h3>
          <button
            onClick={() => setPreview(null)}
            className="text-muted-foreground hover:text-foreground transition text-sm"
          >
            Clear
          </button>
        </div>
        <div className="flex-1 overflow-auto p-4">
          <img src={preview || "/placeholder.svg"} alt="Upload preview" className="w-full h-auto rounded-lg" />
        </div>
      </div>
    )
  }

  return (
    <div
      ref={uploadBoxRef}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      className={`p-8 rounded-lg border-2 border-dashed transition cursor-pointer min-h-[400px] flex flex-col items-center justify-center ${
        isDragging ? "border-accent bg-accent/5" : "border-border/40 bg-card hover:border-accent/50"
      }`}
    >
      <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileInput} className="hidden" />

      <div className="text-center">
        <div className="text-5xl mb-4">🖼️</div>
        <h3 className="text-lg font-semibold text-foreground mb-2">Upload Design Image</h3>
        <p className="text-muted-foreground mb-4">Drag and drop your design image here, or click to select</p>
        <p className="text-xs text-muted-foreground">Supports JPG, PNG, WebP (Max 10MB)</p>
      </div>

      <button
        type="button"
        className="mt-6 px-6 py-2 rounded-lg bg-accent text-accent-foreground font-medium hover:opacity-90 transition"
      >
        Select File
      </button>
    </div>
  )
}
