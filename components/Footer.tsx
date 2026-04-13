export default function Footer() {
  return (
    <footer
      style={{
        borderTop: "1px solid var(--border)",
        padding: "40px 24px",
        marginTop: "80px",
        background: "var(--bg-secondary)",
        position: "relative",
        zIndex: 1,
      }}
    >
      <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: "40px",
            marginBottom: "32px",
          }}
          className="footer-grid"
        >
          {/* Brand */}
          <div>
            <div
              style={{
                fontFamily: "var(--font-display)",
                fontWeight: "800",
                fontSize: "1.2rem",
                marginBottom: "8px",
                letterSpacing: "-0.02em",
              }}
              className="gradient-text"
            >
              ChestPrior
            </div>
            <p
              style={{
                fontSize: "0.82rem",
                color: "var(--text-muted)",
                lineHeight: "1.6",
                maxWidth: "260px",
              }}
            >
              Generative priors & dual feature aggregation for robust chest X-ray classification under data imbalance.
            </p>
          </div>

          {/* Institution */}
          <div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.65rem",
                color: "var(--accent-cyan)",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                marginBottom: "12px",
              }}
            >
              Institution
            </div>
            <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", lineHeight: "1.8" }}>
              KCV Final Project<br />
              Komputasi Cerdas dan Visi<br />
              Institut Teknologi Sepuluh Nopember<br />
              <span style={{ color: "var(--text-muted)" }}>Surabaya, Indonesia</span>
            </p>
          </div>

          {/* Stack */}
          <div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.65rem",
                color: "var(--accent-cyan)",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                marginBottom: "12px",
              }}
            >
              Tech Stack
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {["SDXL", "MedFusion", "DFATB", "FAFN", "ConvNeXtV2", "DINOv2", "MaxViT", "ChestX-ray14"].map((t) => (
                <span
                  key={t}
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.65rem",
                    padding: "2px 8px",
                    background: "var(--bg-elevated)",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                    color: "var(--text-muted)",
                  }}
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div
          style={{
            borderTop: "1px solid var(--border)",
            paddingTop: "20px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--text-muted)" }}>
            © 2026 ChestPrior · KCV Lab Assistant Selection Project
          </span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--text-muted)" }}>
            Built with Next.js · Tailwind · Framer Motion
          </span>
        </div>
      </div>

      <style>{`
        @media (max-width: 768px) {
          .footer-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </footer>
  );
}
