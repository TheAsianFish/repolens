import { ChunkCard } from './ChunkCard'
import type { Chunk } from '../types'

interface Props {
  chunks: Chunk[]
  hasQueried: boolean
}

function labelOrder(label: string | null): number {
  if (!label) return 999
  const m = label.match(/\[(\d+)\]/)
  return m ? parseInt(m[1], 10) : 999
}

export function ChunkList({ chunks, hasQueried }: Props) {
  if (chunks.length === 0 && !hasQueried) {
    return (
      <div
        style={{
          color: 'var(--text-dim)',
          textAlign: 'center',
          padding: '48px 16px',
          fontSize: '13px',
        }}
      >
        Ask a question to see retrieved chunks.
      </div>
    )
  }

  if (chunks.length === 0 && hasQueried) {
    return (
      <div
        style={{
          color: 'var(--text-dim)',
          textAlign: 'center',
          padding: '48px 16px',
          fontSize: '13px',
        }}
      >
        No chunks returned for this query.
      </div>
    )
  }

  const cited = chunks.filter(c => c.is_cited).sort((a, b) => {
    return labelOrder(a.citation_label) - labelOrder(b.citation_label)
  })
  const uncited = chunks.filter(c => !c.is_cited)

  return (
    <div>
      {cited.map(chunk => (
        <ChunkCard
          key={chunk.id}
          chunk={chunk}
          isCited
          citationLabel={chunk.citation_label}
        />
      ))}
      {cited.length > 0 && uncited.length > 0 ? (
        <div
          style={{
            margin: '20px 0 16px',
            borderTop: '1px solid var(--border-subtle)',
            paddingTop: 12,
            color: 'var(--text-dim)',
            fontSize: '11px',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
          }}
        >
          Additional context
        </div>
      ) : null}
      {uncited.map(chunk => (
        <ChunkCard
          key={chunk.id}
          chunk={chunk}
          isCited={false}
          citationLabel={null}
        />
      ))}
    </div>
  )
}
