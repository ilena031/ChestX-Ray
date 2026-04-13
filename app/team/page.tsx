import SectionHeader from "@/components/SectionHeader";

const team = [
  {
    name: "Syahribanun",
    role: "Lead Researcher",
    contrib: "Research design, pipeline development, FE+FA implementation, writeup",
    initials: "SB",
    color: "var(--accent-cyan)",
  },
];

export default function TeamPage() {
  return (
    <div style={{ paddingTop: "100px", paddingBottom: "80px" }}>
      <div style={{ maxWidth: "900px", margin: "0 auto", padding: "0 24px" }}>
        <div style={{ marginBottom: "64px" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--accent-cyan)", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "16px" }}>— Team</div>
          <h1 style={{ fontFamily: "var(--font-display)", fontWeight: "800", fontSize: "clamp(2rem, 5vw, 3.2rem)", letterSpacing: "-0.04em", lineHeight: "1.15", marginBottom: "20px" }}>
            Research Team
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "540px", lineHeight: "1.75" }}>
            KCV Final Project — Komputasi Cerdas dan Visi, Institut Teknologi Sepuluh Nopember (ITS), Surabaya 2026.
          </p>
        </div>

        {/* Team members */}
        <section style={{ marginBottom: "64px" }}>
          <SectionHeader label="Members" title="Researchers" />
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {team.map((member) => (
              <div key={member.name} className="card" style={{ padding: "28px", display: "flex", gap: "24px", alignItems: "flex-start" }}>
                <div
                  style={{
                    width: "56px",
                    height: "56px",
                    borderRadius: "12px",
                    background: `linear-gradient(135deg, ${member.color}30, ${member.color}10)`,
                    border: `1px solid ${member.color}40`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontFamily: "var(--font-mono)",
                    fontWeight: "700",
                    fontSize: "1rem",
                    color: member.color,
                    flexShrink: 0,
                  }}
                >
                  {member.initials}
                </div>
                <div>
                  <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1.15rem", color: "var(--text-primary)", marginBottom: "4px" }}>{member.name}</h3>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: member.color, marginBottom: "10px", letterSpacing: "0.06em" }}>{member.role}</div>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: "1.6" }}>{member.contrib}</p>
                </div>
              </div>
            ))}
          </div>
          <div
            style={{
              marginTop: "20px",
              padding: "20px 24px",
              background: "var(--bg-card)",
              border: "1px dashed var(--border)",
              borderRadius: "10px",
              fontFamily: "var(--font-mono)",
              fontSize: "0.75rem",
              color: "var(--text-muted)",
              textAlign: "center",
            }}
          >
            Update halaman ini dengan menambahkan anggota tim yang sebenarnya di kode
          </div>
        </section>

        {/* Institution */}
        <section style={{ marginBottom: "64px" }}>
          <SectionHeader label="Institution" title="KCV Lab · ITS Surabaya" />
          <div className="card" style={{ padding: "32px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px" }}>
              <div>
                <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1.05rem", marginBottom: "12px", color: "var(--accent-cyan)" }}>
                  Komputasi Cerdas dan Visi (KCV)
                </h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: "1.75" }}>
                  Laboratorium Komputasi Cerdas dan Visi di Departemen Teknik Informatika, Institut Teknologi Sepuluh Nopember (ITS) Surabaya. Penelitian ini merupakan proyek final seleksi asisten lab KCV.
                </p>
              </div>
              <div>
                <h3 style={{ fontFamily: "var(--font-display)", fontWeight: "700", fontSize: "1.05rem", marginBottom: "12px", color: "var(--accent-blue)" }}>
                  Institut Teknologi Sepuluh Nopember
                </h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: "1.75" }}>
                  ITS Surabaya adalah salah satu perguruan tinggi teknik terkemuka di Indonesia, berlokasi di Surabaya, Jawa Timur. Didirikan 1960, ITS dikenal dengan keunggulan di bidang sains dan teknologi.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Acknowledgment */}
        <section>
          <SectionHeader label="Acknowledgment" title="Dataset & References" />
          <div className="card" style={{ padding: "28px" }}>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem", lineHeight: "1.85" }}>
              Dataset ChestX-ray14 disediakan oleh <strong style={{ color: "var(--text-primary)" }}>Wang et al. (2017)</strong> dari NIH Clinical Center — 112.120 gambar frontal-view dari 30.805 pasien dengan 14 label penyakit toraks. Model Medical X-ray Stable Diffusion merupakan fine-tuned variant yang tersedia secara publik. Arsitektur DINOv2, ConvNeXtV2, dan MaxViT tersedia via Hugging Face dan torchvision.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
