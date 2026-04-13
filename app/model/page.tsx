import SectionHeader from "@/components/SectionHeader";

const extractors = [
  {
    name: "Medical X-ray SD (FE+FA)",
    type: "Generative Prior",
    paradigm: "Diffusion-based",
    desc: "UNet denoiser dari Medical X-ray Stable Diffusion digunakan sebagai frozen feature encoder. DFATB + FAFN mengagregasi multi-layer feature maps & attention maps menjadi z₁₂₈.",
    highlights: ["Domain fine-tuned", "UNet encoder", "DFATB + FAFN", "Differential Transformer"],
    color: "var(--accent-cyan)",
    isMain: true,
  },
  {
    name: "SDXL (FE+FA)",
    type: "Generative Prior",
    paradigm: "Diffusion-based",
    desc: "General SDXL UNet sebagai feature encoder dengan pipeline FE+FA yang sama. Pembanding untuk mengukur dampak domain-specific training pada kualitas representasi.",
    highlights: ["General-purpose", "UNet encoder", "DFATB + FAFN", "Same pipeline"],
    color: "var(--accent-blue)",
    isMain: true,
  },
  {
    name: "ConvNeXtV2",
    type: "CNN-based",
    paradigm: "Convolutional",
    desc: "Evolusi CNN modern dengan self-supervised learning (FCMAE). Unggul menangkap fitur lokal dengan efisiensi tinggi. Representasi kuat paradigma berbasis konvolusi.",
    highlights: ["ConvNeXt + FCMAE", "Local features", "Efficient", "Modern CNN"],
    color: "var(--accent-teal)",
    isMain: false,
  },
  {
    name: "DINOv2",
    type: "Transformer-based",
    paradigm: "Vision Transformer",
    desc: "ViT dilatih self-supervised dengan knowledge distillation pada dataset besar tanpa label. Representasi general & transferable, global relation antar region yang kuat.",
    highlights: ["Self-supervised", "Knowledge distill.", "Global attention", "General features"],
    color: "var(--accent-amber)",
    isMain: false,
  },
  {
    name: "MaxViT",
    type: "Hybrid-based",
    paradigm: "Conv + Transformer",
    desc: "Multi-Axis Vision Transformer menggabungkan convolution dengan local window + global grid attention. Menangkap fitur lokal detail sekaligus dependensi global.",
    highlights: ["Multi-axis attn", "Local + global", "Hybrid design", "Efficient blocks"],
    color: "var(--accent-red)",
    isMain: false,
  },
];

export default function ModelPage() {
  return (
    <div style={{ paddingTop: "100px", paddingBottom: "80px" }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 24px" }}>
        <div style={{ marginBottom: "64px" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--accent-cyan)", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "16px" }}>— Model Architecture</div>
          <h1 style={{ fontFamily: "var(--font-display)", fontWeight: "800", fontSize: "clamp(2rem, 5vw, 3.2rem)", letterSpacing: "-0.04em", lineHeight: "1.15", marginBottom: "20px" }}>
            Feature Extractors & Architectures
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "680px", lineHeight: "1.75" }}>
            Lima feature extractor dievaluasi dalam framework yang fair — semua menghasilkan representasi yang masuk ke MLP classifier identik, sehingga perbandingan terisolasi pada kualitas representasi fitur.
          </p>
        </div>

        {/* Main proposed methods */}
        <section style={{ marginBottom: "60px" }}>
          <SectionHeader label="Proposed Methods" title="Generative Prior + FE+FA" subtitle="Dua pendekatan utama menggunakan diffusion model sebagai frozen feature encoder dengan Dual Feature Aggregation pipeline." />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "20px" }}>
            {extractors.filter(e => e.isMain).map((e) => (
              <div key={e.name} className="card" style={{ padding: "32px", borderLeft: `3px solid ${e.color}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "16px" }}>
                  <div>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: e.color, display: "block", marginBottom: "6px", letterSpacing: "0.1em", textTransform: "uppercase" }}>{e.type}</span>
                    <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1.15rem", color: e.color }}>{e.name}</h3>
                  </div>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", padding: "3px 10px", background: `${e.color}15`, border: `1px solid ${e.color}30`, borderRadius: "4px", color: e.color, whiteSpace: "nowrap" }}>{e.paradigm}</span>
                </div>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: "1.75", marginBottom: "16px" }}>{e.desc}</p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {e.highlights.map((h) => (
                    <span key={h} style={{ fontFamily: "var(--font-mono)", fontSize: "0.64rem", padding: "2px 8px", background: `${e.color}10`, border: `1px solid ${e.color}25`, borderRadius: "4px", color: e.color }}>{h}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* FE+FA detail box */}
          <div style={{ padding: "32px", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "12px" }}>
            <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1.1rem", marginBottom: "20px", color: "var(--accent-cyan)" }}>FE+FA Module Detail</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px" }}>
              {[
                { name: "DFATB", fullName: "Dual Feature Aggregation Transformer Block", desc: "Menggabungkan spatial attention dan channel attention secara komplementer dari multi-layer feature maps U-Net." },
                { name: "FAFN", fullName: "Feature Aggregation Feed-Forward Network", desc: "Menangkap informasi spasial non-linear sekaligus mengurangi redundansi channel dari representasi fitur." },
                { name: "Diff. Transformer", fullName: "Differential Denoising Transformer", desc: "Menekan noise pada fitur attention melalui operasi diferensial antara dua attention map independen." },
              ].map((m) => (
                <div key={m.name} style={{ padding: "20px", background: "var(--bg-elevated)", borderRadius: "8px", border: "1px solid var(--border)" }}>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", fontWeight: "700", color: "var(--accent-cyan)", marginBottom: "4px" }}>{m.name}</div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "10px", fontFamily: "var(--font-mono)" }}>{m.fullName}</div>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem", lineHeight: "1.6" }}>{m.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Baseline methods */}
        <section style={{ marginBottom: "60px" }}>
          <SectionHeader label="Baselines" title="Conventional Feature Extractors" subtitle="Tiga paradigma berbeda sebagai pembanding — CNN, ViT, dan Hybrid. Semua menghasilkan representasi ke MLP classifier yang identik." />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px" }}>
            {extractors.filter(e => !e.isMain).map((e) => (
              <div key={e.name} className="card" style={{ padding: "24px" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.64rem", color: e.color, display: "block", marginBottom: "6px", letterSpacing: "0.1em", textTransform: "uppercase" }}>{e.type}</span>
                <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1rem", color: "var(--text-primary)", marginBottom: "8px" }}>{e.name}</h3>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.63rem", padding: "2px 8px", background: `${e.color}12`, border: `1px solid ${e.color}30`, borderRadius: "4px", color: e.color, display: "inline-block", marginBottom: "12px" }}>{e.paradigm}</span>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem", lineHeight: "1.7", marginBottom: "14px" }}>{e.desc}</p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "5px" }}>
                  {e.highlights.map((h) => <span key={h} style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", padding: "2px 7px", background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: "4px", color: "var(--text-muted)" }}>{h}</span>)}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* MLP Classifier */}
        <section>
          <SectionHeader label="Classifier" title="Unified MLP Classifier" />
          <div className="card" style={{ padding: "32px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px" }}>
              <div>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem", lineHeight: "1.8", marginBottom: "16px" }}>
                  Semua feature extractor menghasilkan representasi 128-dimensi (z₁₂₈) yang diumpankan ke MLP classifier yang identik. Ini memastikan evaluasi yang fair dan terisolasi pada kualitas representasi fitur.
                </p>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem", lineHeight: "1.8" }}>
                  Arsitektur MLP yang sama digunakan untuk semua 5 metode di semua 4 skenario, sehingga perbedaan performa murni mencerminkan kemampuan feature extractor.
                </p>
              </div>
              <div style={{ background: "var(--bg-elevated)", borderRadius: "8px", padding: "24px", fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--text-secondary)", lineHeight: "2" }}>
                <div style={{ color: "var(--accent-cyan)", marginBottom: "8px" }}>// MLP Classifier Architecture</div>
                <div>Input: z₁₂₈ [B, 128]</div>
                <div style={{ paddingLeft: "12px", color: "var(--text-muted)" }}>↓ Linear(128, 256)</div>
                <div style={{ paddingLeft: "12px", color: "var(--text-muted)" }}>↓ ReLU + Dropout</div>
                <div style={{ paddingLeft: "12px", color: "var(--text-muted)" }}>↓ Linear(256, 128)</div>
                <div style={{ paddingLeft: "12px", color: "var(--text-muted)" }}>↓ ReLU + Dropout</div>
                <div style={{ paddingLeft: "12px", color: "var(--text-muted)" }}>↓ Linear(128, 6)</div>
                <div>Output: logits [B, 6 classes]</div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
