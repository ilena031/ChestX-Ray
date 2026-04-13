import SectionHeader from "@/components/SectionHeader";

const methods = ["MedSD (FE+FA)", "SDXL (FE+FA)", "ConvNeXtV2", "DINOv2", "MaxViT"];
const methodColors = ["var(--accent-cyan)", "var(--accent-blue)", "var(--accent-teal)", "var(--accent-amber)", "var(--accent-red)"];

const dimensions = [
  {
    aspect: "Paradigm",
    values: ["Generative Prior", "Generative Prior", "Convolutional", "Self-supervised ViT", "Hybrid Conv+ViT"],
  },
  {
    aspect: "Domain",
    values: ["Medical X-ray specific", "General-purpose", "General (fine-tuned)", "General (large-scale)", "General"],
  },
  {
    aspect: "Feature Source",
    values: ["UNet denoiser layers", "UNet denoiser layers", "ConvNeXt blocks", "ViT attention", "Multi-axis blocks"],
  },
  {
    aspect: "Aggregation",
    values: ["DFATB + FAFN + DiffTF", "DFATB + FAFN + DiffTF", "GAP (standard)", "CLS token", "GAP (standard)"],
  },
  {
    aspect: "Output dim",
    values: ["128", "128", "128", "128", "128"],
  },
  {
    aspect: "Pre-training data",
    values: ["Chest X-ray domain", "LAION-5B (general)", "ImageNet-22K", "LVD-142M", "ImageNet-21K"],
  },
  {
    aspect: "Frozen encoder",
    values: ["✓ Yes", "✓ Yes", "✓ Yes (eval)", "✓ Yes (eval)", "✓ Yes (eval)"],
  },
];

const scenarios = ["Balanced\nClean", "Imbalanced\nClean", "Balanced\nCorrupt", "Imbalanced\nCorrupt"];

// Placeholder scores — to be filled with real results
const mockScores = [
  { method: "MedSD (FE+FA)", scores: [78.4, 71.2, 65.8, 59.3] },
  { method: "SDXL (FE+FA)", scores: [76.1, 68.9, 63.2, 57.1] },
  { method: "ConvNeXtV2", scores: [80.2, 72.1, 61.4, 54.8] },
  { method: "DINOv2", scores: [82.5, 74.3, 66.9, 60.2] },
  { method: "MaxViT", scores: [79.8, 71.6, 64.1, 57.9] },
];

const getColor = (val: number, col: number): string => {
  const colVals = mockScores.map((m) => m.scores[col]);
  const max = Math.max(...colVals);
  const min = Math.min(...colVals);
  const norm = (val - min) / (max - min);
  if (norm > 0.8) return "rgba(0,200,180,0.15)";
  if (norm < 0.2) return "rgba(255,82,82,0.1)";
  return "transparent";
};

export default function ComparisonPage() {
  return (
    <div style={{ paddingTop: "100px", paddingBottom: "80px" }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 24px" }}>
        <div style={{ marginBottom: "64px" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--accent-cyan)", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "16px" }}>— Comparison</div>
          <h1 style={{ fontFamily: "var(--font-display)", fontWeight: "800", fontSize: "clamp(2rem, 5vw, 3.2rem)", letterSpacing: "-0.04em", lineHeight: "1.15", marginBottom: "20px" }}>
            Method Comparison
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "640px", lineHeight: "1.75" }}>
            5 feature extractor dibandingkan secara multidimensional — dari karakteristik arsitektur, pendekatan, hingga performa di 4 skenario evaluasi.
          </p>
        </div>

        {/* Architecture Comparison */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader label="Architecture" title="Side-by-Side Architecture Comparison" />
          <div style={{ overflowX: "auto" }}>
            <table style={{ minWidth: "800px" }}>
              <thead>
                <tr>
                  <th style={{ minWidth: "140px" }}>Aspect</th>
                  {methods.map((m, i) => (
                    <th key={m} style={{ color: methodColors[i], minWidth: "160px" }}>{m}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dimensions.map((dim) => (
                  <tr key={dim.aspect}>
                    <td style={{ color: "var(--text-primary)", fontWeight: "500" }}>{dim.aspect}</td>
                    {dim.values.map((v, i) => (
                      <td key={i} style={{ color: dim.aspect === "Output dim" ? methodColors[i] : "var(--text-secondary)" }}>{v}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Performance Heatmap */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader
            label="Performance"
            title="Accuracy Across Scenarios"
            subtitle="Placeholder values — will be updated with actual experimental results. Higher = better, color-coded per column."
          />
          <div className="card" style={{ overflow: "hidden", marginBottom: "16px" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: "12px" }}>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-muted)", letterSpacing: "0.08em" }}>ACCURACY (%) — MLP CLASSIFIER HEAD</span>
              <span style={{ marginLeft: "auto", fontFamily: "var(--font-mono)", fontSize: "0.65rem", padding: "2px 8px", background: "rgba(255,193,77,0.1)", border: "1px solid rgba(255,193,77,0.3)", borderRadius: "4px", color: "var(--accent-amber)" }}>PLACEHOLDER VALUES</span>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ minWidth: "700px" }}>
                <thead>
                  <tr>
                    <th>Method</th>
                    {scenarios.map((s) => (
                      <th key={s}>{s.replace("\n", " · ")}</th>
                    ))}
                    <th>Avg Drop (A→D)</th>
                  </tr>
                </thead>
                <tbody>
                  {mockScores.map((row, ri) => {
                    const drop = row.scores[0] - row.scores[3];
                    return (
                      <tr key={row.method}>
                        <td style={{ color: methodColors[ri], fontWeight: "600" }}>{row.method}</td>
                        {row.scores.map((score, si) => (
                          <td key={si} style={{ background: getColor(score, si), textAlign: "center", color: "var(--text-primary)" }}>
                            {score.toFixed(1)}
                          </td>
                        ))}
                        <td style={{ textAlign: "center", color: drop > 20 ? "var(--accent-red)" : "var(--accent-teal)" }}>
                          −{drop.toFixed(1)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-muted)" }}>
            ✦ Green = best in column · Red = worst in column · All values are placeholder pending experiment completion
          </p>
        </section>

        {/* Tradeoff Analysis */}
        <section style={{ marginBottom: "60px" }}>
          <SectionHeader label="Analysis" title="Tradeoffs & Key Insights" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
            {[
              {
                title: "Generative Priors",
                question: "Apakah domain-specific diffusion model menghasilkan fitur yang lebih robust?",
                pro: "Medical SD memiliki prior domain chest X-ray yang kuat — representasi distribusi patologis yang lebih mendalam",
                con: "Inference lebih lambat dibanding CNN; memerlukan noise injection step tambahan",
                color: "var(--accent-cyan)",
              },
              {
                title: "FE+FA vs Standard GAP",
                question: "Apakah agregasi fitur yang lebih kompleks memberikan keunggulan?",
                pro: "DFATB + FAFN + Differential Transformer secara teori lebih informatif dari GAP biasa",
                con: "Kompleksitas lebih tinggi; perlu validasi apakah keunggulan signifikan di semua skenario",
                color: "var(--accent-teal)",
              },
              {
                title: "Clean vs Corrupt",
                question: "Seberapa jauh performa degradasi saat gambar corrupt?",
                pro: "Model dengan generative prior mungkin lebih robust karena terlatih menghandle distribusi noise",
                con: "Semua model diperkirakan drop signifikan — seberapa besar drop adalah pertanyaan utama",
                color: "var(--accent-amber)",
              },
              {
                title: "Balanced vs Imbalanced",
                question: "Mana yang paling tahan terhadap class imbalance?",
                pro: "ViT-based (DINOv2) cenderung lebih robust terhadap imbalance karena global attention",
                con: "CNN lokal (ConvNeXtV2) mungkin cenderung bias ke majority class dalam skenario imbalanced",
                color: "var(--accent-blue)",
              },
            ].map((item) => (
              <div key={item.title} className="card" style={{ padding: "24px" }}>
                <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1rem", color: item.color, marginBottom: "8px" }}>{item.title}</h3>
                <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--text-muted)", fontStyle: "italic", marginBottom: "14px", lineHeight: "1.5" }}>{item.question}</p>
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <span style={{ color: "var(--accent-teal)", fontSize: "0.8rem", minWidth: "14px" }}>+</span>
                    <span style={{ color: "var(--text-secondary)", fontSize: "0.82rem", lineHeight: "1.6" }}>{item.pro}</span>
                  </div>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <span style={{ color: "var(--accent-red)", fontSize: "0.8rem", minWidth: "14px" }}>−</span>
                    <span style={{ color: "var(--text-secondary)", fontSize: "0.82rem", lineHeight: "1.6" }}>{item.con}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
