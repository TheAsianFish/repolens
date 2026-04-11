export interface Citation {
  label: string
  file_rel_path: string
  file_path: string
  start_line: number
  end_line: number
  name: string
  is_truncated: boolean
}

export interface Chunk {
  id: string
  name: string
  node_type: string
  file_rel_path: string
  file_path: string
  start_line: number
  end_line: number
  source: string
  score: number
  is_cited: boolean
  citation_label: string | null
}
