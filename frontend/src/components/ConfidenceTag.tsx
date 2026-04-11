interface Props {
  confidence: 'high' | 'medium' | 'low'
}

const COLORS = {
  high: { rgb: '34, 197, 94', hex: '#22c55e' },
  medium: { rgb: '245, 158, 11', hex: '#f59e0b' },
  low: { rgb: '239, 68, 68', hex: '#ef4444' },
} as const

export function ConfidenceTag({ confidence }: Props) {
  const c = COLORS[confidence]
  const label = confidence

  return (
    <span
      className="confidence-tag"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '5px',
        fontSize: '11px',
        fontWeight: 600,
        padding: '2px 8px',
        borderRadius: '999px',
        background: `rgba(${c.rgb}, 0.15)`,
        color: c.hex,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
      }}
    >
      <span aria-hidden>●</span>
      {label}
    </span>
  )
}
