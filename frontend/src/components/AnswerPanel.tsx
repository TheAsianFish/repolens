import type { Citation } from '../types'
import { ConfidenceTag } from './ConfidenceTag'

interface Props {
  answer: string
  citations: Citation[]
  confidence: 'high' | 'medium' | 'low'
}

function scrollToChunk(label: string) {
  const el = document.getElementById(`chunk-${label}`)
  el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
}

function renderAnswerWithBadges(text: string) {
  const parts = text.split(/(\[\d+\])/g)
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/)
    if (m) {
      const fullLabel = part
      return (
        <button
          key={`${part}-${i}`}
          type="button"
          onClick={() => scrollToChunk(fullLabel)}
          style={{
            background: 'var(--accent-dim)',
            color: 'var(--accent)',
            borderRadius: 'var(--radius-sm)',
            padding: '0 5px',
            fontSize: '11px',
            cursor: 'pointer',
            fontWeight: 600,
            border: 'none',
            fontFamily: 'inherit',
            verticalAlign: 'baseline',
          }}
          onMouseEnter={e => {
            const t = e.currentTarget
            t.style.background = 'var(--accent)'
            t.style.color = 'white'
          }}
          onMouseLeave={e => {
            const t = e.currentTarget
            t.style.background = 'var(--accent-dim)'
            t.style.color = 'var(--accent)'
          }}
        >
          {part}
        </button>
      )
    }
    return <span key={i}>{part}</span>
  })
}

export function AnswerPanel({ answer, citations, confidence }: Props) {
  return (
    <div
      style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '16px',
        marginTop: '12px',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: 12,
          marginBottom: 12,
        }}
      >
        <span
          style={{
            color: 'var(--text-secondary)',
            fontSize: '11px',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}
        >
          Answer
        </span>
        <ConfidenceTag confidence={confidence} />
      </div>
      <div style={{ color: 'var(--text-primary)', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
        {renderAnswerWithBadges(answer)}
      </div>
      {citations.length > 0 ? (
        <div style={{ marginTop: 20 }}>
          <div
            style={{
              color: 'var(--text-secondary)',
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              marginBottom: 8,
            }}
          >
            Citations
          </div>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {citations.map(c => (
              <li
                key={c.label}
                className="mono"
                style={{
                  fontSize: '12.5px',
                  color: 'var(--text-primary)',
                  marginBottom: 4,
                }}
              >
                {c.label} {c.file_rel_path}:{c.start_line}–{c.end_line} ({c.name})
                {c.is_truncated ? (
                  <span style={{ color: 'var(--text-dim)' }}> [truncated]</span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}
