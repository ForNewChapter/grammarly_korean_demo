import type { TextEdit } from './types';

export function applyEdits(text: string, edits: TextEdit[]): string {
  if (!edits.length) return text;
  const sorted = [...edits].sort((a, b) => b.range.start - a.range.start);
  let out = text;
  for (const edit of sorted) {
    out = `${out.slice(0, edit.range.start)}${edit.replacement}${out.slice(edit.range.end)}`;
  }
  return out;
}
