"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"

export function Navbar() {
  const router = useRouter()

  const handleLogout = () => {
    // TODO: Implement logout
    router.push("/")
  }

  return (
    <nav className="fixed top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-md">
      <div className="mx-auto max-w-6xl px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-accent rounded flex items-center justify-center text-accent-foreground font-bold text-sm">
              C
            </div>
            <span className="font-semibold text-foreground">CodeGen</span>
          </Link>

          <button
            onClick={handleLogout}
            className="px-4 py-2 rounded-lg border border-border text-muted-foreground hover:text-foreground transition"
          >
            Sign Out
          </button>
        </div>
      </div>
    </nav>
  )
}
