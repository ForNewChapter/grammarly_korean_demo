import type { ProtectedSpan, TextEdit } from '../pipeline/types';
import { isProtected } from './protectedSpan';
import type { PhraseRule } from '../worker/runtimeAssets';

export function runPhrasePreNormalizer(
  text: string,
  protectedSpans: ProtectedSpan[],
  phraseMemory: PhraseRule[] = []
): TextEdit[] {
  const edits: TextEdit[] = [];
  const sorted = [...phraseMemory]
    .filter((item) => item.from && item.to)
    .sort((a, b) => b.from.length - a.from.length || b.confidence - a.confidence);

  for (const item of sorted) {
    let from = 0;
    while (from < text.length) {
      const idx = text.indexOf(item.from, from);
      if (idx < 0) break;
      const range = { start: idx, end: idx + item.from.length };
      if (!isProtected(range, protectedSpans)) {
        edits.push({
          stage: 'PHRASE',
          range,
          sourceText: item.from,
          replacement: item.to,
          editType: 'SPELL',
          confidence: item.confidence,
          autoApplicable: item.confidence >= 0.97,
          reasonTag: item.reasonTag,
        });
      }
      from = idx + item.from.length;
    }
  }

  edits.sort((a, b) => a.range.start - b.range.start || b.range.end - a.range.end);

  const nonOverlapping: TextEdit[] = [];
  let lastEnd = -1;
  for (const edit of edits) {
    if (edit.range.start < lastEnd) continue;
    nonOverlapping.push(edit);
    lastEnd = edit.range.end;
  }
  return nonOverlapping;
}
