import SectionHeader from "@/components/SectionHeader";

const noiseTypes = [
  { name: "Brightness (dark)", range: "gamma 0.70 → 0.15", level: "Semakin gelap" },
  { name: "Brightness (bright)", range: "gamma 1.40 → 3.20", level: "Semakin terang" },
  { name: "Gaussian noise", range: "std 5 → 60", level: "Semakin noisy" },
  { name: "Salt & pepper", range: "density 0.01 → 0.15", level: "Semakin corrupt" },
  { name: "Gaussian blur", range: "kernel 3 → 21", level: "Semakin blur" },
  { name: "Motion blur", range: "kernel 5 → 31", level: "Semakin blur" },
  { name: "Occlusion", range: "1–6 patch, 5% → 22%", level: "Semakin tertutup" },
];

const pipelineSteps = [
  { num: "01", title: "VAE Encoding", desc: "Gambar X-ray [B, 3, H, W] dikompresi ke latent space menjadi z₀ [B, 4, h, w] menggunakan Variational Autoencoder.", code: "X-ray → VAE Encoder → z₀", color: "var(--accent-cyan)" },
  { num: "02", title: "Diffusion Noising", desc: "Noise skala kecil ditambahkan ke latent z₀ menghasilkan zₜ. Diperlukan untuk menjembatani gap antara pre-training dan feature extraction.", code: "z₀ + ε → zₜ", color: "var(--accent-blue)" },
  { num: "03", title: "Frozen U-Net Forward Pass", desc: "Latent di-noise diumpankan ke U-Net frozen. Menghasilkan feature maps multi-layer (F*) dan attention maps ({Aᵢ}) dari cross-attention layers.", code: "zₜ → U-Net (frozen) → F*, {Aᵢ}", color: "var(--accent-teal)" },
  { num: "04", title: "DFATB & FAFN", desc: "Feature maps diproses DFATB yang menggabungkan spatial + channel attention secara komplementer. FAFN menangkap informasi non-linear & mengurangi redundansi channel.", code: "F* → DFATB → FAFN → v₁", color: "var(--accent-cyan)" },
  { num: "05", title: "Differential Denoising", desc: "Differential Transformer menekan noise fitur melalui operasi diferensial antara dua attention map independen, memisahkan informasi relevan dari artefak.", code: "{Aᵢ} → Diff. Transformer → u₁", color: "var(--accent-blue)" },
  { num: "06", title: "GAP → Concat → z₁₂₈", desc: "Feature maps & attention maps di-GAP menghasilkan v₁, v₂, u₁. Ketiganya di-concatenate lalu diproyeksi bottleneck menjadi z₁₂₈ untuk MLP classifier.", code: "concat([v₁,v₂,u₁]) → z₁₂₈", color: "var(--accent-teal)" },
];

export default function Methodology() {
  return (
    <div style={{ paddingTop: "100px", paddingBottom: "80px" }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 24px" }}>
        <div style={{ marginBottom: "64px" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--accent-cyan)", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "16px" }}>— Methodology</div>
          <h1 style={{ fontFamily: "var(--font-display)", fontWeight: "800", fontSize: "clamp(2rem, 5vw, 3.2rem)", letterSpacing: "-0.04em", lineHeight: "1.15", marginBottom: "20px" }}>
            Pipeline & Experimental Design
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "640px", lineHeight: "1.75" }}>
            Dual-approach framework menggabungkan generative diffusion priors dengan attention-based feature aggregation, dievaluasi di 4 skenario data klinis yang realistis.
          </p>
        </div>

        {/* 4 Scenarios */}
        <section style={{ marginBottom: "80px" }}>
          <SectionHeader label="Experimental Design" title="4 Evaluation Scenarios" subtitle="Kombinasi dua dimensi: distribusi kelas (balanced/imbalanced) × kualitas gambar (clean/corrupt)." />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            {[
              { id: "Scenario A", title: "Balanced · Clean", desc: "~416 gambar per kelas, tanpa gangguan. Baseline ideal untuk mengukur kemampuan representasi fitur murni.", badge: "Baseline", color: "var(--accent-teal)" },
              { id: "Scenario B", title: "Imbalanced · Clean", desc: "Distribusi tidak seimbang (10:1), gambar bersih. Simulasi dataset medis nyata dengan class imbalance.", badge: "Real-world", color: "var(--accent-amber)" },
              { id: "Scenario C", title: "Balanced · Corrupt", desc: "~416/kelas + 7 jenis noise (3 severity). Uji robustness terhadap degradasi gambar dalam data seimbang.", badge: "Noise", color: "var(--accent-blue)" },
              { id: "Scenario D", title: "Imbalanced · Corrupt", desc: "Worst-case: imbalance + corruption bersamaan. Kondisi paling realistis dan paling menantang.", badge: "Hard", color: "var(--accent-red)" },
            ].map((s) => (
              <div key={s.id} className="card" style={{ padding: "28px", borderTop: `2px solid ${s.color}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--text-muted)" }}>{s.id}</span>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", padding: "2px 8px", background: `${s.color}18`, border: `1px solid ${s.color}40`, borderRadius: "4px", color: s.color }}>{s.badge}</span>
                </div>
                <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1.1rem", marginBottom: "10px" }}>{s.title}</h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: "1.7" }}>{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Noise types */}
        <section style={{ marginBottom: "80px" }}>
          <SectionHeader label="Corruption Design" title="7 Noise Types · 3 Severity Levels" />
          <div className="card" style={{ overflow: "hidden" }}>
            <table>
              <thead><tr><th>Noise Type</th><th>Parameter Range</th><th>Effect</th></tr></thead>
              <tbody>
                {noiseTypes.map((n) => (
                  <tr key={n.name}>
                    <td><strong style={{ color: "var(--text-primary)" }}>{n.name}</strong></td>
                    <td><code>{n.range}</code></td>
                    <td style={{ color: "var(--text-muted)" }}>{n.level}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* FE+FA Pipeline */}
        <section style={{ marginBottom: "80px" }}>
          <SectionHeader label="Core Pipeline" title="Dual Feature Aggregation (FE+FA)" subtitle="6-step pipeline dari gambar X-ray mentah ke representasi 128-dim untuk klasifikasi." />
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {pipelineSteps.map((step) => (
              <div key={step.num} style={{ display: "grid", gridTemplateColumns: "64px 1fr 200px", gap: "20px", alignItems: "center", padding: "20px 24px", background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "10px" }}>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.5rem", fontWeight: "700", color: "var(--border-bright)", lineHeight: "1" }}>{step.num}</div>
                <div>
                  <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "600", fontSize: "1rem", color: step.color, marginBottom: "6px" }}>{step.title}</h3>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem", lineHeight: "1.6" }}>{step.desc}</p>
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--text-muted)", textAlign: "right", lineHeight: "1.5" }}>{step.code}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Generative Prior */}
        <section style={{ marginBottom: "40px" }}>
          <SectionHeader label="Generative Priors" title="Diffusion Models as Feature Extractors" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "20px" }}>
            {[
              { name: "Medical X-ray Stable Diffusion", note: "Primary", desc: "Fine-tuned SD khusus domain chest X-ray. Memiliki pemahaman mendalam terhadap distribusi piksel X-ray, tekstur jaringan paru, dan pola patologis subtle.", tags: ["Domain-specific", "Fine-tuned", "Medical"], color: "var(--accent-cyan)" },
              { name: "Stable Diffusion XL (SDXL)", note: "Comparison", desc: "General-purpose SDXL sebagai pembanding. Mengukur apakah domain-specific training memberikan keunggulan dalam feature extraction medical imaging.", tags: ["General-purpose", "SDXL", "Baseline"], color: "var(--accent-blue)" },
            ].map((m) => (
              <div key={m.name} className="card" style={{ padding: "28px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
                  <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1rem", color: m.color }}>{m.name}</h3>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", padding: "2px 8px", background: `${m.color}15`, border: `1px solid ${m.color}40`, borderRadius: "4px", color: m.color }}>{m.note}</span>
                </div>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: "1.7", marginBottom: "14px" }}>{m.desc}</p>
                <div style={{ display: "flex", gap: "6px" }}>
                  {m.tags.map((t) => <span key={t} style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", padding: "2px 8px", background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: "4px", color: "var(--text-muted)" }}>{t}</span>)}
                </div>
              </div>
            ))}
          </div>
          <div className="card" style={{ padding: "24px", borderColor: "rgba(0,212,255,0.2)", background: "rgba(0,212,255,0.03)" }}>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: "1.7" }}>
              <strong style={{ color: "var(--accent-cyan)" }}>Catatan penting:</strong> Generative prior <em>tidak</em> digunakan untuk menghasilkan gambar baru. UNet denoiser dibekukan (frozen) dan dimanfaatkan sebagai <strong style={{ color: "var(--text-primary)" }}>feature encoder</strong> — representasi internal yang dipelajari selama pre-training digunakan langsung untuk ekstraksi fitur dari gambar X-ray yang ada.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
