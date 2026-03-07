import type { InputMode, TextRange } from '../pipeline/types';

export interface RangeOutput {
  clauseText: string;
  range: TextRange;
}

export function extractRange(text: string, cursor: number, mode: InputMode): RangeOutput {
  if (mode === 'inspect') {
    return { clauseText: text.slice(0, 512), range: { start: 0, end: Math.min(text.length, 512) } };
  }

  const safeCursor = Math.max(0, Math.min(cursor, text.length));
  let before = text.slice(0, safeCursor);
  const separators = ['\n', '.', '?', '!', ',', ';'];
  if (safeCursor > 0 && separators.includes(text[safeCursor - 1])) {
    before = text.slice(0, safeCursor - 1);
  }
  const pivot = separators.map((s) => before.lastIndexOf(s)).reduce((a, b) => Math.max(a, b), -1);
  const start = Math.max(0, pivot + 1);
  const end = safeCursor;
  const raw = text.slice(start, end);
  const clauseText = raw.slice(-128);
  return {
    clauseText,
    range: {
      start: end - clauseText.length,
      end,
    },
  };
}
