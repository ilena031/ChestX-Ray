"use client";
import { useEffect, useRef } from "react";
import Link from "next/link";
import SectionHeader from "@/components/SectionHeader";

const diseases = [
  { code: "No Finding", label: "No Finding", count: 416, imbalance: "1.0×", risk: "Normal", color: "#00c8b4" },
  { code: "Infiltration", label: "Infiltration", count: 417, imbalance: "1.0×", risk: "Moderate", color: "#2979ff" },
  { code: "Effusion", label: "Effusion", count: 416, imbalance: "1.0×", risk: "Moderate", color: "#2979ff" },
  { code: "Atelectasis", label: "Atelectasis", count: 416, imbalance: "1.0×", risk: "Moderate", color: "#ffc14d" },
  { code: "Nodule", label: "Nodule", count: 418, imbalance: "1.0×", risk: "Suspicious", color: "#ffc14d" },
  { code: "Pneumothorax", label: "Pneumothorax", count: 417, imbalance: "1.0×", risk: "Critical", color: "#ff5252" },
];

const imbalancedDiseases = [
  { code: "No Finding", count: 1000, total: 2500, color: "#00c8b4" },
  { code: "Infiltration", count: 600, total: 2500, color: "#2979ff" },
  { code: "Effusion", count: 400, total: 2500, color: "#2979ff" },
  { code: "Atelectasis", count: 250, total: 2500, color: "#ffc14d" },
  { code: "Nodule", count: 150, total: 2500, color: "#ffc14d" },
  { code: "Pneumothorax", count: 100, total: 2500, color: "#ff5252" },
];

const approaches = [
  {
    icon: "⬡",
    title: "Generative Priors",
    desc: "UNet denoiser dari Medical X-ray Stable Diffusion digunakan sebagai feature encoder — bukan untuk generate gambar, melainkan mengekstrak representasi domain-specific dari X-ray.",
    tags: ["SDXL", "MedFusion", "UNet"],
    color: "var(--accent-cyan)",
  },
  {
    icon: "⊕",
    title: "Dual Feature Aggregation",
    desc: "DFATB + FAFN menggabungkan feature maps multi-layer dan attention maps dari U-Net menjadi representasi 128-dim yang informatif melalui spatial & channel attention.",
    tags: ["DFATB", "FAFN", "z₁₂₈"],
    color: "var(--accent-blue)",
  },
  {
    icon: "◈",
    title: "4-Scenario Evaluation",
    desc: "Pengujian komprehensif di 4 kondisi: Balanced, Imbalanced, Corrupt+Balanced, Corrupt+Imbalanced — menggunakan 6 jenis noise dengan 3 severity level.",
    tags: ["Balanced", "Imbalanced", "Corrupt"],
    color: "var(--accent-teal)",
  },
  {
    icon: "⊞",
    title: "Comparative Baselines",
    desc: "Dibandingkan terhadap ConvNeXtV2 (CNN), DINOv2 (ViT), dan MaxViT (Hybrid) — semua dengan MLP classifier yang identik untuk evaluasi fair.",
    tags: ["ConvNeXtV2", "DINOv2", "MaxViT"],
    color: "var(--accent-amber)",
  },
];

export default function Home() {
  return (
    <div style={{ paddingTop: "60px" }}>
      {/* Hero */}
      <section
        className="grid-bg"
        style={{
          minHeight: "92vh",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          textAlign: "center",
          padding: "80px 24px 60px",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Radial glow */}
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: "600px",
            height: "600px",
            borderRadius: "50%",
            background: "radial-gradient(ellipse, rgba(0,102,255,0.08) 0%, transparent 70%)",
            pointerEvents: "none",
          }}
        />
        {/* Xray circles decoration */}
        <div style={{ position: "absolute", top: "10%", left: "5%", opacity: 0.06 }}>
          <svg width="200" height="200" viewBox="0 0 200 200" fill="none">
            <circle cx="100" cy="100" r="90" stroke="#00d4ff" strokeWidth="0.8" strokeDasharray="4 4" />
            <circle cx="100" cy="100" r="60" stroke="#00d4ff" strokeWidth="0.8" />
            <circle cx="100" cy="100" r="30" stroke="#00d4ff" strokeWidth="0.8" strokeDasharray="2 3" />
          </svg>
        </div>
        <div style={{ position: "absolute", bottom: "10%", right: "5%", opacity: 0.05 }}>
          <svg width="280" height="280" viewBox="0 0 280 280" fill="none">
            <rect x="20" y="20" width="240" height="240" stroke="#2979ff" strokeWidth="0.6" strokeDasharray="6 6" />
            <rect x="60" y="60" width="160" height="160" stroke="#2979ff" strokeWidth="0.6" />
            <line x1="0" y1="140" x2="280" y2="140" stroke="#2979ff" strokeWidth="0.4" strokeDasharray="3 6" />
            <line x1="140" y1="0" x2="140" y2="280" stroke="#2979ff" strokeWidth="0.4" strokeDasharray="3 6" />
          </svg>
        </div>

        <div style={{ position: "relative", zIndex: 2, maxWidth: "880px" }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              fontFamily: "var(--font-mono)",
              fontSize: "0.68rem",
              color: "var(--accent-cyan)",
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              marginBottom: "28px",
              padding: "6px 16px",
              border: "1px solid rgba(0,212,255,0.25)",
              borderRadius: "20px",
              background: "rgba(0,212,255,0.05)",
            }}
          >
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--accent-cyan)", display: "inline-block", animation: "pulse-glow 2s infinite" }} />
            KCV Final Project · ITS · 2026
          </div>

          <h1
            style={{
              fontFamily: "var(--font-display)",
              fontWeight: "800",
              fontSize: "clamp(2rem, 5.5vw, 4rem)",
              lineHeight: "1.1",
              letterSpacing: "-0.04em",
              marginBottom: "28px",
              color: "var(--text-primary)",
            }}
          >
            Generative Priors for{" "}
            <span className="gradient-text">Robust Chest X-ray</span>
            <br />Classification
          </h1>

          <p
            style={{
              fontSize: "clamp(0.9rem, 2vw, 1.1rem)",
              color: "var(--text-secondary)",
              maxWidth: "640px",
              margin: "0 auto 40px",
              lineHeight: "1.75",
            }}
          >
            Comparative analysis of generative priors and dual feature aggregation for chest X-ray classification under data imbalance & image corruption — evaluated across 4 clinical-realistic scenarios.
          </p>

          <div style={{ display: "flex", gap: "14px", justifyContent: "center", flexWrap: "wrap" }}>
            <Link
              href="/methodology"
              style={{
                textDecoration: "none",
                padding: "12px 28px",
                background: "linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))",
                borderRadius: "8px",
                fontFamily: "var(--font-mono)",
                fontSize: "0.8rem",
                color: "#fff",
                fontWeight: "700",
                letterSpacing: "0.04em",
                boxShadow: "0 4px 20px rgba(0,212,255,0.2)",
              }}
            >
              Methodology →
            </Link>
            <Link
              href="/results"
              style={{
                textDecoration: "none",
                padding: "12px 28px",
                background: "transparent",
                border: "1px solid var(--border-bright)",
                borderRadius: "8px",
                fontFamily: "var(--font-mono)",
                fontSize: "0.8rem",
                color: "var(--text-secondary)",
                letterSpacing: "0.04em",
              }}
            >
              View Results
            </Link>
          </div>

          {/* Stats row */}
          <div
            style={{
              display: "flex",
              gap: "32px",
              justifyContent: "center",
              marginTop: "60px",
              flexWrap: "wrap",
            }}
          >
            {[
              { val: "112K+", label: "X-ray images" },
              { val: "2,500", label: "Sampled subset" },
              { val: "6", label: "Disease classes" },
              { val: "4", label: "Test scenarios" },
              { val: "4", label: "Feature extractors" },
            ].map(({ val, label }) => (
              <div key={label} style={{ textAlign: "center" }}>
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "1.6rem",
                    fontWeight: "700",
                    color: "var(--accent-cyan)",
                    lineHeight: "1",
                    marginBottom: "4px",
                  }}
                  className="text-glow"
                >
                  {val}
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                  {label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* Abstract */}
      <section style={{ padding: "80px 24px", maxWidth: "1200px", margin: "0 auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.6fr", gap: "64px", alignItems: "start" }} className="two-col">
          <div>
            <SectionHeader label="Abstract" title="The Problem We're Solving" />
            <div
              style={{
                padding: "20px",
                background: "var(--bg-card)",
                border: "1px solid var(--border)",
                borderLeft: "3px solid var(--accent-cyan)",
                borderRadius: "8px",
                fontFamily: "var(--font-mono)",
                fontSize: "0.8rem",
                color: "var(--text-secondary)",
                fontStyle: "italic",
                lineHeight: "1.7",
              }}
            >
              "Seberapa robust sebuah model klasifikasi chest X-ray ketika dihadapkan pada data yang imbalanced, corrupt, atau keduanya sekaligus?"
            </div>
          </div>
          <div>
            <p style={{ color: "var(--text-secondary)", lineHeight: "1.85", fontSize: "0.95rem", marginBottom: "16px" }}>
              Chest X-ray adalah modalitas pencitraan medis paling umum untuk deteksi penyakit toraks. Meski deep learning telah mencapai performa radiologis, model-model tersebut diuji dalam kondisi ideal yang jarang mencerminkan realitas klinis.
            </p>
            <p style={{ color: "var(--text-secondary)", lineHeight: "1.85", fontSize: "0.95rem", marginBottom: "16px" }}>
              Penelitian ini mengeksplorasi dua pendekatan inovatif: <strong style={{ color: "var(--text-primary)" }}>generative prior berbasis diffusion model</strong> (SDXL & Medical X-ray SD) sebagai feature extractor, dan modul <strong style={{ color: "var(--text-primary)" }}>Dual Feature Aggregation (FE+FA)</strong> untuk representasi yang lebih robust.
            </p>
            <p style={{ color: "var(--text-secondary)", lineHeight: "1.85", fontSize: "0.95rem" }}>
              Dataset ChestX-ray14 (112.120 gambar, 30.805 pasien) disampling 2.500 gambar dengan 6 kelas penyakit representatif, dievaluasi di 4 skenario berbasis kombinasi distribusi dan kualitas gambar.
            </p>
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* 4 Key Approaches */}
      <section style={{ padding: "80px 24px", maxWidth: "1200px", margin: "0 auto" }}>
        <SectionHeader
          label="Key Components"
          title="What Makes This Research Different"
          subtitle="Dua pendekatan utama dievaluasi dan dibandingkan terhadap feature extractor konvensional dalam 4 skenario data yang berbeda."
        />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }} className="two-col">
          {approaches.map((a) => (
            <div
              key={a.title}
              className="card"
              style={{ padding: "28px", borderLeft: `2px solid ${a.color}` }}
            >
              <div style={{ fontSize: "1.6rem", marginBottom: "12px" }}>{a.icon}</div>
              <h3
                style={{
                  fontFamily: "var(--font-display)",
                  fontWeight: "700",
                  fontSize: "1.1rem",
                  color: "var(--text-primary)",
                  marginBottom: "10px",
                }}
              >
                {a.title}
              </h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: "1.7", marginBottom: "14px" }}>
                {a.desc}
              </p>
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                {a.tags.map((t) => (
                  <span
                    key={t}
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.65rem",
                      padding: "2px 8px",
                      background: `${a.color}15`,
                      border: `1px solid ${a.color}40`,
                      borderRadius: "4px",
                      color: a.color,
                    }}
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <div className="section-divider" />

      {/* Dataset & Class Imbalance */}
      <section style={{ padding: "80px 24px", maxWidth: "1200px", margin: "0 auto" }}>
        <SectionHeader
          label="Dataset"
          title="ChestX-ray14 — 6 Disease Classes"
          subtitle="2.500 gambar disampling dari 112.120 total, mencakup 6 kelas representatif dengan kondisi balanced dan imbalanced."
        />

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px" }} className="two-col">
          {/* Balanced table */}
          <div className="card" style={{ overflow: "hidden" }}>
            <div
              style={{
                padding: "16px 20px",
                borderBottom: "1px solid var(--border)",
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.7rem",
                  color: "var(--accent-teal)",
                  letterSpacing: "0.1em",
                }}
              >
                BALANCED SCENARIO
              </span>
              <span
                style={{
                  marginLeft: "auto",
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.65rem",
                  padding: "2px 8px",
                  background: "rgba(0,200,180,0.1)",
                  border: "1px solid rgba(0,200,180,0.3)",
                  borderRadius: "4px",
                  color: "var(--accent-teal)",
                }}
              >
                ~416/class
              </span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Samples</th>
                  <th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {diseases.map((d) => (
                  <tr key={d.code}>
                    <td>
                      <code style={{ color: d.color, fontSize: "0.78rem" }}>{d.code}</code>
                    </td>
                    <td>{d.count}</td>
                    <td>
                      <span
                        style={{
                          color: d.color,
                          fontFamily: "var(--font-mono)",
                          fontSize: "0.7rem",
                        }}
                      >
                        {d.risk}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Imbalanced visualization */}
          <div className="card" style={{ padding: "20px" }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.7rem",
                color: "var(--accent-amber)",
                letterSpacing: "0.1em",
                marginBottom: "20px",
              }}
            >
              IMBALANCED SCENARIO — Distribution
            </div>
            {imbalancedDiseases.map((d) => (
              <div key={d.code} style={{ marginBottom: "14px" }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: "5px",
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.72rem",
                  }}
                >
                  <span style={{ color: d.color }}>{d.code}</span>
                  <span style={{ color: "var(--text-muted)" }}>{d.count}</span>
                </div>
                <div
                  style={{
                    height: "6px",
                    background: "var(--border)",
                    borderRadius: "3px",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${(d.count / d.total) * 100}%`,
                      background: d.color,
                      borderRadius: "3px",
                      opacity: 0.8,
                    }}
                  />
                </div>
              </div>
            ))}
            <p
              style={{
                fontSize: "0.75rem",
                color: "var(--text-muted)",
                marginTop: "16px",
                fontFamily: "var(--font-mono)",
              }}
            >
              * Imbalanced ratio: 10:1 (No Finding vs Pneumothorax)
            </p>
          </div>
        </div>
      </section>

      <div className="section-divider" />

      {/* Research Objectives */}
      <section style={{ padding: "80px 24px", maxWidth: "1200px", margin: "0 auto" }}>
        <SectionHeader label="Objectives" title="Research Goals" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "20px" }} className="three-col">
          {[
            {
              num: "01",
              title: "Evaluate Generative Priors",
              desc: "Bandingkan SDXL dan Medical X-ray SD sebagai feature extractor dengan pendekatan konvensional (CNN, ViT, Hybrid) di semua skenario.",
            },
            {
              num: "02",
              title: "Validate FE+FA Module",
              desc: "Ukur kontribusi Dual Feature Aggregation (DFATB + FAFN + Differential Transformer) terhadap representasi fitur yang lebih robust.",
            },
            {
              num: "03",
              title: "Robustness Under Corruption",
              desc: "Evaluasi degradasi performa model saat data corrupt (6 noise types, 3 severity levels) dalam kondisi balanced dan imbalanced.",
            },
          ].map((obj) => (
            <div key={obj.num} className="card" style={{ padding: "28px" }}>
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "2rem",
                  fontWeight: "700",
                  color: "var(--border-bright)",
                  marginBottom: "16px",
                  lineHeight: "1",
                }}
              >
                {obj.num}
              </div>
              <h3
                style={{
                  fontFamily: "var(--font-display)",
                  fontWeight: "600",
                  fontSize: "1rem",
                  color: "var(--text-primary)",
                  marginBottom: "10px",
                }}
              >
                {obj.title}
              </h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.83rem", lineHeight: "1.7" }}>
                {obj.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section
        style={{
          padding: "80px 24px",
          maxWidth: "1200px",
          margin: "0 auto",
          textAlign: "center",
        }}
      >
        <div
          style={{
            padding: "60px",
            background: "linear-gradient(135deg, rgba(14,25,38,0.9), rgba(18,32,48,0.9))",
            border: "1px solid var(--border-bright)",
            borderRadius: "16px",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: "radial-gradient(ellipse at 50% 0%, rgba(0,212,255,0.06) 0%, transparent 60%)",
            }}
          />
          <h2
            style={{
              fontFamily: "var(--font-display)",
              fontWeight: "700",
              fontSize: "clamp(1.4rem, 3vw, 2rem)",
              marginBottom: "16px",
              position: "relative",
            }}
          >
            Explore the Full Research
          </h2>
          <p
            style={{
              color: "var(--text-secondary)",
              marginBottom: "32px",
              fontSize: "0.95rem",
              position: "relative",
            }}
          >
            Dive into methodology, model architectures, comparison, and results.
          </p>
          <div
            style={{
              display: "flex",
              gap: "12px",
              justifyContent: "center",
              flexWrap: "wrap",
              position: "relative",
            }}
          >
            {[
              { href: "/methodology", label: "Methodology" },
              { href: "/model", label: "Model Architecture" },
              { href: "/comparison", label: "Comparison" },
              { href: "/results", label: "Results" },
              { href: "/team", label: "Team" },
            ].map((l) => (
              <Link
                key={l.href}
                href={l.href}
                style={{
                  textDecoration: "none",
                  padding: "9px 20px",
                  border: "1px solid var(--border-bright)",
                  borderRadius: "6px",
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.75rem",
                  color: "var(--text-secondary)",
                  transition: "all 0.2s",
                }}
              >
                {l.label}
              </Link>
            ))}
          </div>
        </div>
      </section>

      <style>{`
        @media (max-width: 768px) {
          .two-col { grid-template-columns: 1fr !important; }
          .three-col { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}
