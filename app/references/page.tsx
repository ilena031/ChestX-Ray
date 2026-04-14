const refs = [
  { id:"1",  authors:"Wang, X., Peng, Y., Lu, L., et al.", year:"2017", title:"ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks", venue:"CVPR 2017", tag:"Dataset" },
  { id:"2",  authors:"Rajpurkar, P., Irvin, J., et al.", year:"2017", title:"CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning", venue:"arXiv:1711.05225", tag:"Baseline" },
  { id:"3",  authors:"Rombach, R., Blattmann, A., et al.", year:"2022", title:"High-Resolution Image Synthesis with Latent Diffusion Models", venue:"CVPR 2022", tag:"Generative" },
  { id:"4",  authors:"Podell, D., English, Z., et al.", year:"2023", title:"SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis", venue:"arXiv:2307.01952", tag:"Generative" },
  { id:"5",  authors:"Woo, S., Debnath, S., et al.", year:"2023", title:"ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders", venue:"CVPR 2023", tag:"Extractor" },
  { id:"6",  authors:"Oquab, M., Darcet, T., et al.", year:"2023", title:"DINOv2: Learning Robust Visual Features without Supervision", venue:"TMLR 2023", tag:"Extractor" },
  { id:"7",  authors:"Tu, Z., Talebi, H., et al.", year:"2022", title:"MaxViT: Multi-Axis Vision Transformer", venue:"ECCV 2022", tag:"Extractor" },
  { id:"8",  authors:"Baranchuk, D., Rubachev, I., et al.", year:"2022", title:"Label-Efficient Semantic Segmentation with Diffusion Models", venue:"ICLR 2022", tag:"Diffusion Features" },
  { id:"9",  authors:"Ye, T., Dong, L., et al.", year:"2024", title:"Differential Transformer", venue:"arXiv:2410.05258", tag:"Architecture" },
  { id:"10", authors:"Dosovitskiy, A., Beyer, L., et al.", year:"2021", title:"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale", venue:"ICLR 2021", tag:"Architecture" },
  { id:"11", authors:"Vaswani, A., Shazeer, N., et al.", year:"2017", title:"Attention Is All You Need", venue:"NeurIPS 2017", tag:"Architecture" },
  { id:"12", authors:"He, K., Chen, X., et al.", year:"2022", title:"Masked Autoencoders Are Scalable Vision Learners", venue:"CVPR 2022", tag:"Pre-training" },
];

const tagColor: Record<string, string> = {
  Dataset:"var(--accent-2)", Baseline:"var(--ink-mid)", Generative:"var(--accent)",
  Extractor:"var(--ink-mid)", "Diffusion Features":"var(--accent)", Architecture:"var(--ink-mid)", "Pre-training":"var(--accent-2)",
};

export default function ReferencesPage() {
  return (
    <main style={{ paddingTop: "56px" }}>
      <div style={{ background: "var(--ink)", padding: "52px 40px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "#6a5f50", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "14px" }}>A for admin · References</div>
          <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "clamp(2rem,4vw,3rem)", color: "var(--paper)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Bibliography
          </h1>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "72px 40px" }}>
        <div style={{ display: "flex", flexDirection: "column" }}>
          {refs.map((r, i) => (
            <div key={r.id} style={{
              display: "grid", gridTemplateColumns: "40px 1fr",
              gap: "20px", padding: "20px 0",
              borderBottom: i < refs.length - 1 ? "1px solid var(--rule)" : "none",
            }}>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-faint)", paddingTop: "3px" }}>[{r.id}]</span>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "12px", marginBottom: "4px" }}>
                  <h3 style={{ fontFamily: "var(--font-serif)", fontWeight: 600, fontSize: "0.95rem", color: "var(--ink)", lineHeight: "1.4" }}>{r.title}</h3>
                  <span className="chip chip-ink" style={{ whiteSpace: "nowrap", fontSize: "0.58rem", flexShrink: 0 }}>{r.tag}</span>
                </div>
                <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--ink-light)", marginBottom: "2px" }}>{r.authors}</p>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: tagColor[r.tag] }}>{r.venue} · {r.year}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
