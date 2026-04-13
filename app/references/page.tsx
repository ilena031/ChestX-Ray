import SectionHeader from "@/components/SectionHeader";

const references = [
  {
    id: "1",
    authors: "Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M.",
    year: "2017",
    title: "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks",
    venue: "CVPR 2017",
    tag: "Dataset",
    color: "var(--accent-cyan)",
  },
  {
    id: "2",
    authors: "Rajpurkar, P., Irvin, J., Ball, R. L., et al.",
    year: "2017",
    title: "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning",
    venue: "arXiv:1711.05225",
    tag: "Baseline",
    color: "var(--accent-blue)",
  },
  {
    id: "3",
    authors: "Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.",
    year: "2022",
    title: "High-Resolution Image Synthesis with Latent Diffusion Models",
    venue: "CVPR 2022",
    tag: "Generative",
    color: "var(--accent-teal)",
  },
  {
    id: "4",
    authors: "Podell, D., English, Z., Lacey, K., et al.",
    year: "2023",
    title: "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis",
    venue: "arXiv:2307.01952",
    tag: "Generative",
    color: "var(--accent-teal)",
  },
  {
    id: "5",
    authors: "Woo, S., Debnath, S., Hu, R., et al.",
    year: "2023",
    title: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders",
    venue: "CVPR 2023",
    tag: "Feature Extractor",
    color: "var(--accent-amber)",
  },
  {
    id: "6",
    authors: "Oquab, M., Darcet, T., Moutakanni, T., et al.",
    year: "2023",
    title: "DINOv2: Learning Robust Visual Features without Supervision",
    venue: "TMLR 2023",
    tag: "Feature Extractor",
    color: "var(--accent-amber)",
  },
  {
    id: "7",
    authors: "Tu, Z., Talebi, H., Zhang, H., et al.",
    year: "2022",
    title: "MaxViT: Multi-Axis Vision Transformer",
    venue: "ECCV 2022",
    tag: "Feature Extractor",
    color: "var(--accent-amber)",
  },
  {
    id: "8",
    authors: "Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.",
    year: "2021",
    title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    venue: "ICLR 2021",
    tag: "Architecture",
    color: "var(--accent-blue)",
  },
  {
    id: "9",
    authors: "Baranchuk, D., Rubachev, I., Voynov, A., Khrulkov, V., & Babenko, A.",
    year: "2022",
    title: "Label-Efficient Semantic Segmentation with Diffusion Models",
    venue: "ICLR 2022",
    tag: "Diffusion Features",
    color: "var(--accent-cyan)",
  },
  {
    id: "10",
    authors: "Vaswani, A., Shazeer, N., Parmar, N., et al.",
    year: "2017",
    title: "Attention Is All You Need",
    venue: "NeurIPS 2017",
    tag: "Architecture",
    color: "var(--accent-blue)",
  },
  {
    id: "11",
    authors: "Ye, T., Dong, L., Xia, Y., et al.",
    year: "2024",
    title: "Differential Transformer",
    venue: "arXiv:2410.05258",
    tag: "Architecture",
    color: "var(--accent-red)",
  },
  {
    id: "12",
    authors: "He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R.",
    year: "2022",
    title: "Masked Autoencoders Are Scalable Vision Learners",
    venue: "CVPR 2022",
    tag: "Pre-training",
    color: "var(--accent-teal)",
  },
];

const tagColors: Record<string, string> = {
  Dataset: "var(--accent-cyan)",
  Baseline: "var(--accent-blue)",
  Generative: "var(--accent-teal)",
  "Feature Extractor": "var(--accent-amber)",
  Architecture: "var(--accent-blue)",
  "Diffusion Features": "var(--accent-cyan)",
  "Pre-training": "var(--accent-teal)",
};

const allTags = [...new Set(references.map((r) => r.tag))];

export default function ReferencesPage() {
  return (
    <div style={{ paddingTop: "100px", paddingBottom: "80px" }}>
      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
        {/* Header */}
        <div style={{ marginBottom: "64px" }}>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.7rem",
              color: "var(--accent-cyan)",
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              marginBottom: "16px",
            }}
          >
            — References
          </div>
          <h1
            style={{
              fontFamily: "var(--font-display)",
              fontWeight: "800",
              fontSize: "clamp(2rem, 5vw, 3.2rem)",
              letterSpacing: "-0.04em",
              lineHeight: "1.15",
              marginBottom: "20px",
            }}
          >
            Bibliography
          </h1>
          <p
            style={{
              color: "var(--text-secondary)",
              fontSize: "1.05rem",
              maxWidth: "600px",
              lineHeight: "1.75",
            }}
          >
            {references.length} references across datasets, generative models, feature extractors, and architectural foundations.
          </p>
        </div>

        {/* Tag legend */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "8px",
            marginBottom: "40px",
            padding: "20px 24px",
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            borderRadius: "10px",
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.68rem",
              color: "var(--text-muted)",
              letterSpacing: "0.08em",
              marginRight: "8px",
              alignSelf: "center",
            }}
          >
            CATEGORIES:
          </span>
          {allTags.map((tag) => (
            <span
              key={tag}
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.66rem",
                padding: "3px 10px",
                background: `${tagColors[tag]}12`,
                border: `1px solid ${tagColors[tag]}35`,
                borderRadius: "4px",
                color: tagColors[tag],
              }}
            >
              {tag}
            </span>
          ))}
        </div>

        {/* Reference list */}
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          {references.map((ref) => (
            <div
              key={ref.id}
              style={{
                display: "grid",
                gridTemplateColumns: "40px 1fr",
                gap: "16px",
                padding: "20px 24px",
                background: "var(--bg-card)",
                border: "1px solid var(--border)",
                borderRadius: "10px",
                transition: "border-color 0.2s",
              }}
              className="ref-item"
            >
              {/* Number */}
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.72rem",
                  color: "var(--text-muted)",
                  paddingTop: "2px",
                  textAlign: "right",
                }}
              >
                [{ref.id}]
              </div>

              {/* Content */}
              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    gap: "12px",
                    marginBottom: "6px",
                  }}
                >
                  <p
                    style={{
                      fontFamily: "var(--font-display)",
                      fontWeight: "600",
                      fontSize: "0.92rem",
                      color: "var(--text-primary)",
                      lineHeight: "1.4",
                    }}
                  >
                    {ref.title}
                  </p>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.62rem",
                      padding: "2px 8px",
                      background: `${tagColors[ref.tag]}12`,
                      border: `1px solid ${tagColors[ref.tag]}30`,
                      borderRadius: "4px",
                      color: tagColors[ref.tag],
                      whiteSpace: "nowrap",
                      flexShrink: 0,
                    }}
                  >
                    {ref.tag}
                  </span>
                </div>
                <p
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.72rem",
                    color: "var(--text-secondary)",
                    marginBottom: "4px",
                  }}
                >
                  {ref.authors}
                </p>
                <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.68rem",
                      color: ref.color,
                    }}
                  >
                    {ref.venue}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.66rem",
                      color: "var(--text-muted)",
                    }}
                  >
                    {ref.year}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        <style>{`
          .ref-item:hover { border-color: var(--border-bright) !important; }
        `}</style>
      </div>
    </div>
  );
}
