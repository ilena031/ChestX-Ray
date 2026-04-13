interface SectionHeaderProps {
  label: string;
  title: string;
  subtitle?: string;
}

export default function SectionHeader({ label, title, subtitle }: SectionHeaderProps) {
  return (
    <div style={{ marginBottom: "48px" }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.68rem",
          color: "var(--accent-cyan)",
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          marginBottom: "12px",
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: "24px",
            height: "1px",
            background: "var(--accent-cyan)",
          }}
        />
        {label}
      </div>
      <h2
        style={{
          fontFamily: "var(--font-display)",
          fontWeight: "700",
          fontSize: "clamp(1.6rem, 3vw, 2.4rem)",
          letterSpacing: "-0.03em",
          color: "var(--text-primary)",
          lineHeight: "1.2",
          marginBottom: subtitle ? "16px" : 0,
        }}
      >
        {title}
      </h2>
      {subtitle && (
        <p
          style={{
            color: "var(--text-secondary)",
            fontSize: "1rem",
            maxWidth: "640px",
            lineHeight: "1.7",
          }}
        >
          {subtitle}
        </p>
      )}
    </div>
  );
}
