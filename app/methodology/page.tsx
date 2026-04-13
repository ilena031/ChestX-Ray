import SectionHeader from "@/components/SectionHeader";

const scenarios = [
  { id: "A", label: "Balanced · Clean",      desc: "~416/kelas, tanpa gangguan. Lower-bound ideal untuk mengukur kemampuan representasi fitur murni.",       tag: "Baseline",   col: "var(--accent-2)" },
  { id: "B", label: "Imbalanced · Clean",    desc: "Distribusi 10:1, gambar bersih. Simulasi dataset medis nyata dengan class imbalance tanpa degradasi.",   tag: "Imbalanced", col: "var(--ink-mid)" },
  { id: "C", label: "Balanced · Corrupt",    desc: "~416/kelas + 7 jenis noise (3 severity). Uji robustness terhadap degradasi gambar pada data seimbang.", tag: "Corrupt",    col: "var(--accent)" },
  { id: "D", label: "Imbalanced · Corrupt",  desc: "Worst-case — imbalance + corruption bersamaan. Kondisi paling realistis dan paling menantang.",          tag: "Hard",       col: "#b22222" },
];

const noiseTypes = [
  { name: "Brightness (dark)",   range: "gamma 0.70 → 0.15" },
  { name: "Brightness (bright)", range: "gamma 1.40 → 3.20" },
  { name: "Gaussian noise",      range: "std 5 → 60 (0–255 scale)" },
  { name: "Salt & pepper",       range: "density 0.01 → 0.15" },
  { name: "Gaussian blur",       range: "kernel 3 → 21" },
  { name: "Motion blur",         range: "kernel 5 → 31" },
  { name: "Occlusion",           range: "1–6 patches, 5%–22% area" },
];

const steps = [
  { n:"01", title:"VAE Encoding",              eq:"X-ray → VAE → z₀ [B,4,h,w]",                 desc:"Gambar dikompresi ke latent space menggunakan Variational Autoencoder." },
  { n:"02", title:"Diffusion Noising",          eq:"z₀ + ε → zₜ",                                desc:"Noise kecil ditambahkan untuk menjembatani gap antara pre-training dan feature extraction." },
  { n:"03", title:"Frozen U-Net Forward Pass", eq:"zₜ → U-Net (frozen) → F*, {Aᵢ}",            desc:"Feature maps multi-layer (F*) dan attention maps ({Aᵢ}) diekstrak dari U-Net yang dibekukan." },
  { n:"04", title:"DFATB & FAFN",              eq:"F* → DFATB → FAFN → v₁",                    desc:"Spatial + channel attention diproses secara komplementer; redundansi channel dikurangi." },
  { n:"05", title:"Differential Denoising",    eq:"{Aᵢ} → Diff. Transformer → u₁",             desc:"Noise pada attention maps ditekan melalui operasi diferensial dua attention map independen." },
  { n:"06", title:"Bottleneck → z₁₂₈",        eq:"concat([v₁,v₂,u₁]) → FC → z₁₂₈ [B,128]",   desc:"Ketiga vektor digabungkan dan diproyeksikan ke representasi 128-dim untuk MLP classifier." },
];

export default function Methodology() {
  return (
    <main style={{ paddingTop: "56px" }}>
      {/* Page title band */}
      <div style={{ background: "var(--ink)", padding: "52px 40px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "#6a5f50", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "14px" }}>
            ChestPrior · Methodology
          </div>
          <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "clamp(2rem,4vw,3rem)", color: "var(--paper)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Pipeline &amp; Experimental Design
          </h1>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "72px 40px" }}>

        {/* 4 Scenarios */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="1." label="Experimental Design" title="Four Evaluation Scenarios" subtitle="Kombinasi dua dimensi: distribusi kelas (balanced/imbalanced) × kualitas gambar (clean/corrupt)." />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            {scenarios.map((s) => (
              <div key={s.id} className="card" style={{ padding: "28px", borderTop: `3px solid ${s.col}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <span style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.1rem", color: "var(--ink)" }}>
                    Scenario {s.id}
                  </span>
                  <span className="chip chip-ink">{s.tag}</span>
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: s.col, fontWeight: 500, marginBottom: "10px" }}>{s.label}</div>
                <p style={{ fontSize: "0.85rem", color: "var(--ink-light)", lineHeight: "1.7" }}>{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* Noise types */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="2." label="Corruption Design" title="7 Noise Types · 3 Severity Levels" subtitle="Diterapkan sintetis untuk mensimulasikan degradasi kualitas yang lazim ditemui secara klinis." />
          <div className="card" style={{ overflow: "hidden" }}>
            <table>
              <thead><tr><th>#</th><th>Noise Type</th><th>Parameter Range (Severity 1→3)</th></tr></thead>
              <tbody>
                {noiseTypes.map((n, i) => (
                  <tr key={n.name}>
                    <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-faint)" }}>{String(i + 1).padStart(2, "0")}</td>
                    <td style={{ fontWeight: 500, color: "var(--ink)" }}>{n.name}</td>
                    <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.78rem" }}>{n.range}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* FE+FA Pipeline */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="3." label="Core Pipeline" title="Dual Feature Aggregation (FE+FA)" subtitle="Enam tahap dari gambar X-ray mentah hingga representasi 128-dim siap klasifikasi." />
          <div style={{ position: "relative" }}>
            {/* Vertical line */}
            <div style={{ position: "absolute", left: "23px", top: "16px", bottom: "16px", width: "1px", background: "var(--rule)" }} />
            <div style={{ display: "flex", flexDirection: "column", gap: "0" }}>
              {steps.map((s, i) => (
                <div key={s.n} style={{ display: "grid", gridTemplateColumns: "48px 1fr", gap: "24px", paddingBottom: "28px" }}>
                  {/* Number bubble */}
                  <div style={{
                    width: "46px", height: "46px", borderRadius: "50%",
                    background: i < 2 ? "var(--accent)" : i < 4 ? "var(--ink)" : "var(--accent-2)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontFamily: "var(--font-mono)", fontSize: "0.7rem", fontWeight: 500,
                    color: "#fff", flexShrink: 0, position: "relative", zIndex: 1,
                  }}>{s.n}</div>
                  <div style={{ paddingTop: "8px" }}>
                    <div style={{ fontFamily: "var(--font-serif)", fontWeight: 600, fontSize: "1rem", color: "var(--ink)", marginBottom: "4px" }}>{s.title}</div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--accent)", marginBottom: "6px" }}>{s.eq}</div>
                    <p style={{ fontSize: "0.85rem", color: "var(--ink-light)", lineHeight: "1.6" }}>{s.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* Generative Prior note */}
        <section>
          <SectionHeader index="4." label="Key Concept" title="Generative Prior as Feature Encoder" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
            {[
              { name: "Medical X-ray Stable Diffusion", role: "Primary", desc: "Fine-tuned SD pada domain chest X-ray. Memiliki prior yang lebih mendalam terhadap distribusi intensitas piksel X-ray, tekstur jaringan paru, dan pola patologis subtle." },
              { name: "Stable Diffusion XL",            role: "Comparison", desc: "General-purpose SDXL. Digunakan sebagai pembanding untuk mengukur apakah domain-specific training memberikan keunggulan dalam medical feature extraction." },
            ].map((m) => (
              <div key={m.name} className="card" style={{ padding: "24px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                  <h3 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1rem", color: "var(--ink)" }}>{m.name}</h3>
                  <span className={m.role === "Primary" ? "chip chip-accent" : "chip chip-ink"}>{m.role}</span>
                </div>
                <p style={{ fontSize: "0.84rem", color: "var(--ink-light)", lineHeight: "1.7" }}>{m.desc}</p>
              </div>
            ))}
          </div>
          <div style={{ marginTop: "20px", padding: "20px 24px", background: "var(--accent-muted)", border: "1px solid var(--accent-light)", borderRadius: "4px" }}>
            <p style={{ fontSize: "0.85rem", color: "var(--ink-mid)", lineHeight: "1.7" }}>
              <strong>Penting:</strong> Generative prior tidak digunakan untuk <em>menghasilkan</em> gambar baru. U-Net denoiser dibekukan (frozen) dan berfungsi sebagai <strong>feature encoder</strong> — representasi internal dari pre-training dimanfaatkan langsung untuk mengekstrak fitur dari gambar X-ray yang sudah ada.
            </p>
          </div>
        </section>
      </div>

      <style>{`@media (max-width: 768px) { .two-col { grid-template-columns: 1fr !important; } }`}</style>
    </main>
  );
}
