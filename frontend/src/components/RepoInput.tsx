interface Props {
  repoPath: string
  onRepoPathChange: (path: string) => void
  onIndex: () => void
  isIndexing: boolean
  indexStatus: string | null
}

export function RepoInput({
  repoPath,
  onRepoPathChange,
  onIndex,
  isIndexing,
  indexStatus,
}: Props) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'stretch' }}>
        <input
          type="text"
          value={repoPath}
          onChange={e => onRepoPathChange(e.target.value)}
          placeholder="/path/to/repository"
          disabled={isIndexing}
          style={{
            flex: 1,
            minWidth: 0,
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            color: 'var(--text-primary)',
            padding: '8px 12px',
            width: '100%',
            fontFamily: 'inherit',
            fontSize: '13px',
            outline: 'none',
          }}
          onFocus={e => {
            e.target.style.borderColor = 'var(--accent)'
          }}
          onBlur={e => {
            e.target.style.borderColor = 'var(--border)'
          }}
        />
        <button
          type="button"
          onClick={() => void onIndex()}
          disabled={isIndexing || !repoPath.trim()}
          style={{
            background: 'var(--accent)',
            color: 'white',
            border: 'none',
            borderRadius: 'var(--radius)',
            padding: '8px 16px',
            cursor: isIndexing ? 'not-allowed' : 'pointer',
            fontWeight: 600,
            whiteSpace: 'nowrap',
            opacity: isIndexing ? 0.5 : 1,
          }}
          onMouseEnter={e => {
            if (!isIndexing) (e.target as HTMLButtonElement).style.opacity = '0.85'
          }}
          onMouseLeave={e => {
            (e.target as HTMLButtonElement).style.opacity = isIndexing ? '0.5' : '1'
          }}
        >
          {isIndexing ? 'Indexing...' : 'Index'}
        </button>
      </div>
      {indexStatus ? (
        <p
          style={{
            marginTop: 8,
            fontSize: '12px',
            color: 'var(--text-dim)',
          }}
        >
          {indexStatus}
        </p>
      ) : null}
    </div>
  )
}
