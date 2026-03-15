import type { TextEdit, TextRange } from '../pipeline/types';

export interface HighlightMarker {
  range: TextRange;
  tone: 'issue' | 'suggestion' | 'auto';
}

export interface AppliedPreview {
  text: string;
  markers: HighlightMarker[];
}

function sortEdits(edits: TextEdit[]): TextEdit[] {
  return [...edits].sort((left, right) => {
    if (left.range.start !== right.range.start) return left.range.start - right.range.start;
    return left.range.end - right.range.end;
  });
}

export function applyEditsWithPreview(text: string, edits: TextEdit[], tone: HighlightMarker['tone']): AppliedPreview {
  if (!edits.length) {
    return { text, markers: [] };
  }

  const sorted = sortEdits(edits);
  const chunks: string[] = [];
  const markers: HighlightMarker[] = [];
  let cursor = 0;

  for (const edit of sorted) {
    const { start, end } = edit.range;
    if (start < cursor) continue;

    chunks.push(text.slice(cursor, start));
    const markerStart = chunks.join('').length;
    chunks.push(edit.replacement);
    const markerEnd = chunks.join('').length;
    markers.push({
      range: { start: markerStart, end: markerEnd },
      tone,
    });
    cursor = end;
  }

  chunks.push(text.slice(cursor));
  return {
    text: chunks.join(''),
    markers,
  };
}

export function renderHighlightedText(text: string, markers: HighlightMarker[]): JSX.Element {
  if (!text) return <span>-</span>;
  if (!markers.length) return <span>{text}</span>;

  const ordered = [...markers].sort((left, right) => left.range.start - right.range.start);
  const nodes: JSX.Element[] = [];
  let cursor = 0;

  for (let index = 0; index < ordered.length; index += 1) {
    const marker = ordered[index];
    const { start, end } = marker.range;
    if (start > cursor) {
      nodes.push(<span key={`plain-${index}`}>{text.slice(cursor, start)}</span>);
    }

    const className =
      marker.tone === 'issue'
        ? 'highlight-issue'
        : marker.tone === 'suggestion'
          ? 'highlight-suggestion'
          : 'highlight-auto';

    nodes.push(
      <span key={`mark-${index}`} className={className}>
        {text.slice(start, end)}
      </span>
    );
    cursor = end;
  }

  if (cursor < text.length) {
    nodes.push(<span key="plain-tail">{text.slice(cursor)}</span>);
  }

  return <>{nodes}</>;
}
