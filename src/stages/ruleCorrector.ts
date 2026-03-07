import type { ProtectedSpan, TextEdit } from '../pipeline/types';
import { isProtected } from './protectedSpan';

const COMMON: Array<{ from: string; to: string; reasonTag: string; confidence: number }> = [
  { from: '되요', to: '돼요', reasonTag: 'common_misspelling', confidence: 0.99 },
  { from: '왠', to: '웬', reasonTag: 'common_misspelling', confidence: 0.97 },
  { from: '삿어요', to: '샀어요', reasonTag: 'common_misspelling', confidence: 0.99 },
  { from: '잇어요', to: '있어요', reasonTag: 'common_misspelling', confidence: 0.96 },
];

export function runRuleCorrector(text: string, protectedSpans: ProtectedSpan[]): TextEdit[] {
  const edits: TextEdit[] = [];
  for (const rule of COMMON) {
    let from = 0;
    while (from < text.length) {
      const idx = text.indexOf(rule.from, from);
      if (idx < 0) break;
      const range = { start: idx, end: idx + rule.from.length };
      if (!isProtected(range, protectedSpans)) {
        edits.push({
          stage: 'RULE',
          range,
          sourceText: rule.from,
          replacement: rule.to,
          editType: 'SPELL',
          confidence: rule.confidence,
          autoApplicable: true,
          reasonTag: rule.reasonTag,
        });
      }
      from = idx + rule.from.length;
    }
  }
  return edits;
}
