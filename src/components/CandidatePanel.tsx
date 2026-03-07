import type { TextEdit } from '../pipeline/types';

interface CandidatePanelProps {
  edits: TextEdit[];
  baseText?: string;
}

function labelForEdit(edit: TextEdit): string {
  if (edit.editType === 'SPACE_INSERT' || edit.editType === 'SPACE_DELETE') return '띄어쓰기';
  if (edit.editType === 'SPELL') return '철자';
  if (edit.editType === 'OPEN_REPLACE') return '문맥 제안';
  return edit.editType;
}

function spacingPreview(text: string, edit: TextEdit): string {
  if (edit.editType === 'SPACE_INSERT') {
    return `${text.slice(0, edit.range.start)}|${text.slice(edit.range.start)} → ${text.slice(0, edit.range.start)} ${text.slice(edit.range.start)}`;
  }
  if (edit.editType === 'SPACE_DELETE') {
    return `${text} → ${text.slice(0, edit.range.start)}${text.slice(edit.range.end)}`;
  }
  return `${edit.sourceText || '∅'} → ${edit.replacement || '∅'}`;
}

export function CandidatePanel({ edits, baseText = '' }: CandidatePanelProps): JSX.Element {
  const ordered = [...edits].sort((left, right) => left.range.start - right.range.start);

  return (
    <section className="panel">
      <h2>제안에 포함된 수정</h2>
      <div className="suggestion-card-stack compact-stack">
        {ordered.length ? ordered.map((edit, index) => (
          <article key={`${edit.stage}-${index}`} className="suggestion-card compact-card">
            <div className="suggestion-card-head">
              <div className="suggestion-badges">
                <span className={`pill ${edit.autoApplicable ? 'ok' : 'warn'}`}>
                  {edit.autoApplicable ? '안전 교정' : '문맥 제안'}
                </span>
                <span className="pill muted">{labelForEdit(edit)}</span>
              </div>
              <code>{edit.editType === 'SPACE_INSERT' || edit.editType === 'SPACE_DELETE' ? spacingPreview(baseText, edit) : `${edit.sourceText || '∅'} → ${edit.replacement || '∅'}`}</code>
            </div>
            <div className="suggestion-meta">
              <span>위치: {edit.range.start}-{edit.range.end}</span>
              <span>신뢰도: {Math.round(edit.confidence * 100)}%</span>
            </div>
          </article>
        )) : <div className="box inline">수정 제안이 없습니다.</div>}
      </div>
    </section>
  );
}
