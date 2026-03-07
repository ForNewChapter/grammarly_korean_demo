import type { ProtectedSpan } from '../pipeline/types';
import { isProtected } from './protectedSpan';

export interface TaggedToken {
  token: string;
  label:
    | 'KEEP'
    | 'DIRECT_SPELL_FIX'
    | 'SPACE_FIX'
    | 'JOSA_FIX'
    | 'EOMI_FIX'
    | 'PUNCT_FIX'
    | 'OPEN_REPLACE'
    | 'DELETE'
    | 'MERGE'
    | 'SPLIT';
  confidence: number;
  range: { start: number; end: number };
}

export function tokenizeWithRanges(text: string): Array<{ token: string; start: number; end: number }> {
  const out: Array<{ token: string; start: number; end: number }> = [];
  const re = /\S+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    out.push({ token: m[0], start: m.index, end: m.index + m[0].length });
  }
  return out;
}

function shouldOpenReplace(token: string): boolean {
  if (['낫다', '낫아', '안되', '되요', '할수'].includes(token)) return true;
  if (/^낫(아|어|았|었|으면|으니|지)/.test(token)) return true;
  if (/^됬/.test(token)) return true;
  return false;
}

export function runEditTagger(text: string, protectedSpans: ProtectedSpan[]): TaggedToken[] {
  const tokens = tokenizeWithRanges(text);
  return tokens.map((t) => {
    const range = { start: t.start, end: t.end };
    if (isProtected(range, protectedSpans)) {
      return { token: t.token, label: 'KEEP', confidence: 1, range };
    }
    if (shouldOpenReplace(t.token)) {
      return { token: t.token, label: 'OPEN_REPLACE', confidence: 0.9, range };
    }
    return { token: t.token, label: 'KEEP', confidence: 0.98, range };
  });
}
