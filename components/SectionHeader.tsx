interface Props {
  index?: string;
  label: string;
  title: string;
  subtitle?: string;
}
export default function SectionHeader({ index, label, title, subtitle }: Props) {
  return (
    <div style={{ marginBottom: "40px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "14px" }}>
        {index && (
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--ink-faint)", minWidth: "28px" }}>
            {index}
          </span>
        )}
        <span style={{ height: "1px", flex: 1, background: "var(--rule)", maxWidth: "40px" }} />
        <span style={{
          fontFamily: "var(--font-mono)", fontSize: "0.65rem",
          letterSpacing: "0.14em", textTransform: "uppercase",
          color: "var(--accent)",
        }}>{label}</span>
      </div>
      <h2 style={{
        fontFamily: "var(--font-serif)", fontWeight: 700,
        fontSize: "clamp(1.5rem, 2.8vw, 2.2rem)",
        color: "var(--ink)", lineHeight: "1.2",
        letterSpacing: "-0.02em",
        marginBottom: subtitle ? "14px" : 0,
      }}>{title}</h2>
      {subtitle && (
        <p style={{ color: "var(--ink-light)", fontSize: "0.95rem", maxWidth: "580px", lineHeight: "1.7" }}>
          {subtitle}
        </p>
      )}
    </div>
  );
}
