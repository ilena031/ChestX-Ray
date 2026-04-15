"use client";

export default function TeamPage() {
  const team = [
    { name: "Syahribanun", role: "Lead Researcher", contrib: "Research design · pipeline development · FE+FA implementation · writeup", initials: "S", photo: "/team/syahribanun.jpeg" },
    { name: "Ahmad Naufal Farras", role: "Researcher", contrib: "Classification model development · Feature Extraction module implementation · Model & web deployment", initials: "A", photo: "/team/farras.jpg" },
  ];
  return (
    <main style={{ paddingTop: "56px" }}>
      <div style={{ background: "var(--ink)", padding: "52px 40px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "#6a5f50", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "14px" }}>A for admin · Team</div>
          <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "clamp(2rem,4vw,3rem)", color: "var(--paper)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Research Team
          </h1>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "72px 40px" }}>

        {/* Members */}
        <section style={{ marginBottom: "72px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {team.map((m) => (
              <div key={m.name} className="card" style={{ padding: "32px", display: "grid", gridTemplateColumns: "88px 1fr", gap: "28px", alignItems: "center" }}>
                {/* Photo placeholder — replace src with actual photo */}
                <div style={{
                  width: "88px", height: "88px", borderRadius: "50%",
                  overflow: "hidden", border: "3px solid var(--rule)",
                  background: "var(--ink)", display: "flex", alignItems: "center", justifyContent: "center",
                  flexShrink: 0,
                }}>
                  <img
                    src={m.photo}
                    alt={m.name}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    onError={(e) => {
                      // Fallback to initials if photo not found
                      const target = e.currentTarget;
                      target.style.display = "none";
                      const parent = target.parentElement;
                      if (parent) {
                        parent.innerHTML = `<span style="font-family:var(--font-serif);font-weight:900;font-size:1.6rem;color:var(--paper)">${m.initials}</span>`;
                      }
                    }}
                  />
                </div>
                <div>
                  <h3 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.3rem", color: "var(--ink)", marginBottom: "4px" }}>{m.name}</h3>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--accent)", letterSpacing: "0.06em", marginBottom: "10px", fontWeight: 500 }}>{m.role}</div>
                  <p style={{ fontSize: "0.88rem", color: "var(--ink-light)", lineHeight: "1.7", fontWeight: 400 }}>{m.contrib}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* Institution */}
        <section style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "40px" }}>
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "var(--accent)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "14px" }}>Institution</div>
            <h2 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.3rem", color: "var(--ink)", marginBottom: "12px", letterSpacing: "-0.01em" }}>
              Institut Teknologi Sepuluh Nopember
            </h2>
            <p style={{ fontSize: "0.88rem", color: "var(--ink-light)", lineHeight: "1.8" }}>
              Departemen Teknik Informatika<br />
              Laboratorium Komputasi Cerdas dan Visi (KCV)<br />
              Surabaya, Jawa Timur, Indonesia
            </p>
          </div>
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "var(--accent)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "14px" }}>About This Project</div>
            <p style={{ fontSize: "0.88rem", color: "var(--ink-light)", lineHeight: "1.8" }}>
              Proyek final seleksi asisten lab KCV — menggabungkan penelitian generative model, computer vision, dan medical imaging untuk menjawab pertanyaan robustness yang belum banyak dieksplorasi dalam literatur.
            </p>
          </div>
        </section>
      </div>

      <style>{`@media (max-width: 768px) { section[style*="grid-template-columns: 1fr 1fr"] { grid-template-columns: 1fr !important; } }`}</style>
    </main>
  );
}
