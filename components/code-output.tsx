"use client"

import { useState } from "react"
import gsap from "gsap"
import { useEffect, useRef } from "react"

interface CodeOutputProps {
  code: {
    html: string
    css: string
    react: string
  }
}

type CodeTab = "html" | "css" | "react"

// Simple syntax highlighter without external dependencies
function SyntaxHighlightedCode({ code, language }: { code: string; language: string }) {
  const highlightCode = (code: string, lang: string): string => {
    let highlighted = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")

    if (lang === "jsx" || lang === "html") {
      // Highlight HTML tags
      highlighted = highlighted.replace(/&lt;(\/?[\w]+)[^&]*?&gt;/g, '<span class="text-blue-400">&lt;$1&gt;</span>')
      // Highlight attributes
      highlighted = highlighted.replace(/(\w+)=/g, '<span class="text-cyan-400">$1</span>=')
      // Highlight attribute values
      highlighted = highlighted.replace(/="([^"]*)"/g, '="<span class="text-green-400">$1</span>"')
      // Highlight JSX expressions
      highlighted = highlighted.replace(/{([^}]*)}/g, '{<span class="text-yellow-400">$1</span>}')
    } else if (lang === "css") {
      // Highlight CSS properties
      highlighted = highlighted.replace(/([a-z-]+)(?=\s*:)/g, '<span class="text-cyan-400">$1</span>')
      // Highlight CSS values
      highlighted = highlighted.replace(/:\s*([^;]+);/g, ': <span class="text-green-400">$1</span>;')
      // Highlight selectors
      highlighted = highlighted.replace(/^([^{]+){/gm, '<span class="text-blue-400">$1</span>{')
    }

    return highlighted
  }

  const lines = code.split("\n")
  const highlightedLines = lines.map((line, i) => {
    const highlighted = highlightCode(line, language)
    return (
      <div key={i} className="flex">
        <span className="inline-block w-8 text-right pr-4 text-muted-foreground select-none">{i + 1}</span>
        <span dangerouslySetInnerHTML={{ __html: highlighted }} />
      </div>
    )
  })

  return <div className="font-mono text-sm">{highlightedLines}</div>
}

export function CodeOutput({ code }: CodeOutputProps) {
  const [activeTab, setActiveTab] = useState<CodeTab>("react")
  const [copiedTab, setCopiedTab] = useState<CodeTab | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const copyButtonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (containerRef.current) {
      gsap.fromTo(containerRef.current, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.5, ease: "power2.out" })
    }
  }, [code])

  const tabs: { id: CodeTab; label: string }[] = [
    { id: "html", label: "HTML" },
    { id: "css", label: "CSS" },
    { id: "react", label: "React" },
  ]

  const getCode = (): string => {
    return code[activeTab]
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(getCode())
      setCopiedTab(activeTab)

      if (copyButtonRef.current) {
        gsap.to(copyButtonRef.current, {
          scale: 1.05,
          duration: 0.2,
          yoyo: true,
          repeat: 1,
        })
      }

      setTimeout(() => setCopiedTab(null), 2000)
    } catch (err) {
      console.error("Failed to copy:", err)
    }
  }

  const handleDownload = () => {
    const ext = activeTab === "react" ? "tsx" : activeTab
    const filename = `generated-${activeTab}.${ext}`
    const element = document.createElement("a")
    element.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(getCode()))
    element.setAttribute("download", filename)
    element.style.display = "none"
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  return (
    <div ref={containerRef} className="rounded-lg border border-border/40 bg-card overflow-hidden flex flex-col h-full">
      {/* Tabs */}
      <div className="border-b border-border/40 flex items-center gap-0">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-3 font-mono text-sm font-medium border-b-2 transition ${
              activeTab === tab.id
                ? "border-accent text-accent"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Code Display */}
      <div className="flex-1 overflow-auto p-4 bg-background">
        <SyntaxHighlightedCode code={getCode()} language={activeTab === "react" ? "jsx" : activeTab} />
      </div>

      {/* Actions */}
      <div className="border-t border-border/40 p-4 flex gap-2 flex-wrap">
        <button
          ref={copyButtonRef}
          onClick={handleCopy}
          className="px-4 py-2 rounded-lg border border-border hover:bg-secondary transition text-sm font-medium text-foreground"
        >
          {copiedTab === activeTab ? "✓ Copied" : "Copy Code"}
        </button>
        <button
          onClick={handleDownload}
          className="px-4 py-2 rounded-lg bg-accent/10 border border-accent/30 hover:bg-accent/20 transition text-sm font-medium text-accent"
        >
          Download
        </button>
      </div>
    </div>
  )
}
