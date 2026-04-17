"use client"

import Link from "next/link"
import { useEffect, useRef } from "react"
import gsap from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"

gsap.registerPlugin(ScrollTrigger)

export default function Home() {
  const heroRef = useRef<HTMLDivElement>(null)
  const titleRef = useRef<HTMLHeadingElement>(null)
  const subtitleRef = useRef<HTMLParagraphElement>(null)
  const ctaRef = useRef<HTMLDivElement>(null)
  const featuresRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Hero entrance animation
    const tl = gsap.timeline()

    tl.fromTo(titleRef.current, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" })
      .fromTo(
        subtitleRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.6, ease: "power2.out" },
        "-=0.4",
      )
      .fromTo(ctaRef.current, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.6, ease: "power2.out" }, "-=0.3")

    // Feature cards scroll animation
    const cards = document.querySelectorAll(".feature-card")
    cards.forEach((card, index) => {
      gsap.fromTo(
        card,
        { opacity: 0, y: 40 },
        {
          opacity: 1,
          y: 0,
          duration: 0.6,
          delay: index * 0.1,
          scrollTrigger: {
            trigger: card,
            start: "top 80%",
            toggleActions: "play none none none",
          },
        },
      )
    })

    return () => {
      ScrollTrigger.getAll().forEach((trigger) => trigger.kill())
    }
  }, [])

  return (
    <div ref={heroRef} className="min-h-screen w-full dark bg-background">
      {/* Navigation */}
      <nav className="fixed top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-md">
        <div className="mx-auto max-w-6xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-accent rounded flex items-center justify-center text-accent-foreground font-bold text-sm">
                C
              </div>
              <span className="font-semibold text-foreground">CodeGen</span>
            </div>
            <div className="hidden md:flex gap-8 items-center">
              <a href="#features" className="text-muted-foreground hover:text-foreground transition">
                Features
              </a>
              <a href="#" className="text-muted-foreground hover:text-foreground transition">
                Docs
              </a>
              <Link
                href="/login"
                className="px-6 py-2 rounded-full bg-accent text-accent-foreground font-medium hover:opacity-90 transition"
              >
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <div className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-4xl text-center">
          <div className="mb-6 inline-block px-3 py-1 rounded-full bg-secondary text-muted-foreground text-sm font-medium">
            AI-Powered Design to Code
          </div>

          <h1
            ref={titleRef}
            className="text-5xl sm:text-6xl lg:text-7xl font-bold text-foreground mb-6 text-balance leading-tight"
          >
            Design Images to Production Code
          </h1>

          <p ref={subtitleRef} className="text-lg sm:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto text-balance">
            Upload a design screenshot and instantly get clean, production-ready React code with HTML, CSS, and
            component variants.
          </p>

          <div ref={ctaRef} className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link
              href="/upload"
              className="px-8 py-3 rounded-full bg-accent text-accent-foreground font-semibold hover:opacity-90 transition"
            >
              Start Generating
            </Link>
          </div>
        </div>
      </div>

      {/* Features */}
      <div ref={featuresRef} id="features" className="py-20 px-4 sm:px-6 lg:px-8 border-t border-border/40">
        <div className="mx-auto max-w-6xl">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">Powerful Features for Developers</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Everything you need to turn designs into code in seconds
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                icon: "🎨",
                title: "Design to Code",
                description: "Upload any design image and get clean, semantic HTML and CSS instantly.",
              },
              {
                icon: "⚛️",
                title: "React Components",
                description: "Automatic React component generation with proper props and state management.",
              },
              {
                icon: "🎯",
                title: "Syntax Highlighting",
                description: "Beautiful code display with syntax highlighting for easy reading and copying.",
              },
              {
                icon: "📋",
                title: "Multiple Formats",
                description: "Export as HTML, CSS, React components, or raw JSX code.",
              },
              {
                icon: "⚡",
                title: "Live Preview",
                description: "See your generated code rendered live in an isolated preview container.",
              },
              {
                icon: "💾",
                title: "Download & Copy",
                description: "Copy individual code blocks or download complete files instantly.",
              },
            ].map((feature, i) => (
              <div
                key={i}
                className="feature-card p-6 rounded-lg border border-border/40 bg-card hover:border-accent/50 transition group"
              >
                <div className="text-3xl mb-3">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-foreground mb-2">{feature.title}</h3>
                <p className="text-muted-foreground text-sm">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-20 px-4 sm:px-6 lg:px-8 border-t border-border/40">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-6">
            Ready to Generate Your First Component?
          </h2>
          <p className="text-muted-foreground mb-8 text-balance">
            Sign up now and start turning design images into production-ready code.
          </p>
          <Link
            href="/upload"
            className="inline-block px-8 py-3 rounded-full bg-accent text-accent-foreground font-semibold hover:opacity-90 transition"
          >
            Get Started Now
          </Link>
        </div>
      </div>
    </div>
  )
}
