import SectionHeader from "@/components/SectionHeader";

export default function ResultsPage() {
  return (
    <div style={{ paddingTop: "100px", paddingBottom: "80px" }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 24px" }}>
        <div style={{ marginBottom: "64px" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--accent-cyan)", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "16px" }}>— Results</div>
          <h1 style={{ fontFamily: "var(--font-display)", fontWeight: "800", fontSize: "clamp(2rem, 5vw, 3.2rem)", letterSpacing: "-0.04em", lineHeight: "1.15", marginBottom: "20px" }}>
            Experimental Results
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "640px", lineHeight: "1.75" }}>
            Evaluasi komparatif 5 feature extractor di 4 skenario data. Hasil diukur dengan accuracy, F1-score, dan per-class metrics.
          </p>
        </div>

        {/* Coming soon banner */}
        <div
          style={{
            padding: "48px",
            background: "var(--bg-card)",
            border: "1px solid var(--border-bright)",
            borderRadius: "16px",
            textAlign: "center",
            marginBottom: "48px",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse at 50% 0%, rgba(0,212,255,0.05) 0%, transparent 60%)", pointerEvents: "none" }} />
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              fontFamily: "var(--font-mono)",
              fontSize: "0.68rem",
              color: "var(--accent-amber)",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              marginBottom: "20px",
              padding: "6px 16px",
              border: "1px solid rgba(255,193,77,0.3)",
              borderRadius: "20px",
              background: "rgba(255,193,77,0.06)",
            }}
          >
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--accent-amber)", display: "inline-block", animation: "pulse-glow 2s infinite" }} />
            Experiments in Progress
          </div>
          <h2 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1.6rem", marginBottom: "12px" }}>
            Results Coming Soon
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", maxWidth: "480px", margin: "0 auto", lineHeight: "1.7" }}>
            Eksperimen masih berjalan. Halaman ini akan diisi dengan hasil aktual setelah semua skenario selesai dievaluasi.
          </p>
        </div>

        {/* Metrics to be reported */}
        <section style={{ marginBottom: "60px" }}>
          <SectionHeader label="Evaluation Metrics" title="What Will Be Reported" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px" }}>
            {[
              { metric: "Accuracy", desc: "Overall classification accuracy across 6 disease classes per skenario.", icon: "◎" },
              { metric: "Macro F1-Score", desc: "Unweighted F1 untuk mengakomodasi class imbalance secara adil.", icon: "⊕" },
              { metric: "Per-class F1", desc: "F1 per kelas: No Finding, Infiltration, Effusion, Atelectasis, Nodule, Pneumothorax.", icon: "⊞" },
              { metric: "Robustness Drop", desc: "Degradasi performa dari Scenario A (baseline) ke Scenario D (worst-case).", icon: "⬇" },
              { metric: "Imbalance Sensitivity", desc: "Perbandingan performa Balanced vs Imbalanced untuk mengukur ketahanan terhadap class skew.", icon: "⚖" },
              { metric: "Noise Resilience", desc: "Performa per severity level (1/2/3) untuk setiap jenis noise corruption.", icon: "◈" },
            ].map((m) => (
              <div key={m.metric} className="card" style={{ padding: "24px" }}>
                <div style={{ fontSize: "1.4rem", marginBottom: "10px", color: "var(--accent-cyan)" }}>{m.icon}</div>
                <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "0.95rem", marginBottom: "8px", color: "var(--text-primary)" }}>{m.metric}</h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem", lineHeight: "1.6" }}>{m.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Expected structure */}
        <section>
          <SectionHeader label="Result Structure" title="How Results Will Be Organized" />
          <div className="card" style={{ overflow: "hidden" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--border)" }}>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-muted)", letterSpacing: "0.08em" }}>EXPECTED RESULT TABLE STRUCTURE</span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Scen. A (Bal+Clean)</th>
                  <th>Scen. B (Imbal+Clean)</th>
                  <th>Scen. C (Bal+Corrupt)</th>
                  <th>Scen. D (Imbal+Corrupt)</th>
                </tr>
              </thead>
              <tbody>
                {["MedSD (FE+FA)", "SDXL (FE+FA)", "ConvNeXtV2", "DINOv2", "MaxViT"].map((m, i) => (
                  <tr key={m}>
                    <td style={{ color: ["var(--accent-cyan)", "var(--accent-blue)", "var(--accent-teal)", "var(--accent-amber)", "var(--accent-red)"][i] }}>{m}</td>
                    {[0, 1, 2, 3].map((s) => (
                      <td key={s} style={{ textAlign: "center", color: "var(--text-muted)", fontStyle: "italic" }}>TBD</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ marginTop: "12px", fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-muted)" }}>
            * Values reported as Accuracy (%) / Macro F1 (%). Update halaman ini setelah eksperimen selesai.
          </p>
        </section>
      </div>
    </div>
  );
}
