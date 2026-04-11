interface Props {
  question: string
  onQuestionChange: (q: string) => void
  onSubmit: () => void
  isQuerying: boolean
}

export function QueryBox({
  question,
  onQuestionChange,
  onSubmit,
  isQuerying,
}: Props) {
  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey && question.trim() && !isQuerying) {
      e.preventDefault()
      onSubmit()
    }
  }

  return (
    <div style={{ marginBottom: 16 }}>
      <textarea
        value={question}
        onChange={e => onQuestionChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything about this codebase..."
        disabled={isQuerying}
        rows={3}
        style={{
          width: '100%',
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          color: 'var(--text-primary)',
          padding: '8px 12px',
          fontFamily: 'inherit',
          fontSize: '13px',
          resize: 'vertical',
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
        onClick={() => question.trim() && onSubmit()}
        disabled={isQuerying || !question.trim()}
        style={{
          width: '100%',
          marginTop: 8,
          background: 'var(--accent)',
          color: 'white',
          border: 'none',
          borderRadius: 'var(--radius)',
          padding: '8px 16px',
          cursor: isQuerying ? 'not-allowed' : 'pointer',
          fontWeight: 600,
          opacity: isQuerying ? 0.5 : 1,
        }}
        onMouseEnter={e => {
          if (!isQuerying) (e.target as HTMLButtonElement).style.opacity = '0.85'
        }}
        onMouseLeave={e => {
          (e.target as HTMLButtonElement).style.opacity = isQuerying ? '0.5' : '1'
        }}
      >
        {isQuerying ? 'Asking...' : 'Ask'}
      </button>
    </div>
  )
}
