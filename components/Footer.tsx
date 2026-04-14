export default function Footer() {
  return (
    <footer style={{
      borderTop: "2px solid var(--ink)",
      padding: "48px 40px",
      background: "var(--ink)",
      color: "var(--paper-dark)",
      marginTop: "120px",
    }}>
      <div style={{ maxWidth: "1100px", margin: "0 auto", display: "grid", gridTemplateColumns: "2fr 1fr 1fr", gap: "48px" }}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.3rem", color: "var(--paper)", marginBottom: "12px" }}>
            A for admin
          </div>
          <p style={{ fontSize: "0.83rem", lineHeight: "1.7", color: "#a09080", maxWidth: "320px" }}>
            Comparative analysis of generative priors and dual feature aggregation for robust chest X-ray classification under data imbalance.
          </p>
        </div>
        <div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", letterSpacing: "0.12em", color: "#6a5f50", textTransform: "uppercase", marginBottom: "14px" }}>
            Institution
          </div>
          <p style={{ fontSize: "0.8rem", lineHeight: "1.9", color: "#a09080" }}>
            Komputasi Cerdas dan Visi<br/>
            Teknik Informatika<br/>
            ITS Surabaya
          </p>
        </div>

      </div>
      <div style={{ maxWidth: "1100px", margin: "32px auto 0", paddingTop: "24px", borderTop: "1px solid #2e2820", fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "#6a5f50", display: "flex", justifyContent: "space-between" }}>
        <span>© 2026 A for admin · KCV Lab Selection Project</span>
        <span>ChestX-ray14 · Wang et al., 2017</span>
      </div>
    </footer>
  );
}
