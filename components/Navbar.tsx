"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const navLinks = [
  { href: "/", label: "Overview" },
  { href: "/methodology", label: "Methodology" },
  { href: "/model", label: "Model" },
  { href: "/comparison", label: "Comparison" },
  { href: "/results", label: "Results" },
  { href: "/team", label: "Team" },
  { href: "/references", label: "References" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handler);
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <nav
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        padding: "0 24px",
        height: "60px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: scrolled
          ? "rgba(6, 11, 17, 0.95)"
          : "rgba(6, 11, 17, 0.7)",
        borderBottom: scrolled
          ? "1px solid var(--border-bright)"
          : "1px solid var(--border)",
        backdropFilter: "blur(20px)",
        transition: "all 0.3s ease",
      }}
    >
      {/* Logo */}
      <Link href="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: "10px" }}>
        <div
          style={{
            width: "28px",
            height: "28px",
            borderRadius: "6px",
            background: "linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "12px",
            fontWeight: "700",
            color: "#fff",
            fontFamily: "var(--font-mono)",
          }}
        >
          CP
        </div>
        <span
          style={{
            fontFamily: "var(--font-display)",
            fontWeight: "700",
            fontSize: "1rem",
            color: "var(--text-primary)",
            letterSpacing: "-0.02em",
          }}
        >
          ChestPrior
        </span>
      </Link>

      {/* Desktop links */}
      <div style={{ display: "flex", gap: "4px", alignItems: "center" }} className="hidden-mobile">
        {navLinks.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            style={{
              textDecoration: "none",
              padding: "5px 12px",
              borderRadius: "6px",
              fontFamily: "var(--font-mono)",
              fontSize: "0.72rem",
              letterSpacing: "0.04em",
              color:
                pathname === link.href
                  ? "var(--accent-cyan)"
                  : "var(--text-secondary)",
              background:
                pathname === link.href
                  ? "rgba(0,212,255,0.08)"
                  : "transparent",
              border:
                pathname === link.href
                  ? "1px solid rgba(0,212,255,0.2)"
                  : "1px solid transparent",
              transition: "all 0.2s ease",
            }}
          >
            {link.label}
          </Link>
        ))}
      </div>

      {/* Badge */}
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.65rem",
          color: "var(--text-muted)",
          letterSpacing: "0.06em",
        }}
        className="hidden-mobile"
      >
        KCV · ITS · 2026
      </div>

      <style>{`
        @media (max-width: 768px) {
          .hidden-mobile { display: none !important; }
        }
      `}</style>
    </nav>
  );
}
