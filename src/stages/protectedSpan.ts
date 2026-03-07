import type { ProtectedSpan, TextRange } from '../pipeline/types';

const PATTERNS: Array<{ kind: ProtectedSpan['kind']; regex: RegExp }> = [
  { kind: 'URL', regex: /https?:\/\/[^\s]+/g },
  { kind: 'EMAIL', regex: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi },
  { kind: 'PHONE', regex: /\b\d{2,4}-\d{3,4}-\d{4}\b/g },
  { kind: 'HASHTAG', regex: /#[\w가-힣]+/g },
  { kind: 'MENTION', regex: /@[\w가-힣._-]+/g },
  { kind: 'NUMBER', regex: /\b\d+(?:[.,]\d+)?\b/g },
];

const USER_DICT = ['온디바이스', '맞춤법', '교정기'];
const DOMAIN_ENTITIES = ['AirPods Pro 2', 'RTX 4060', 'ChatGPT'];

function overlap(a: TextRange, b: TextRange): boolean {
  return a.start < b.end && b.start < a.end;
}

export function detectProtectedSpans(text: string): { protectedSpans: ProtectedSpan[]; maskedText: string } {
  const found: Array<Omit<ProtectedSpan, 'placeholder'>> = [];

  for (const p of PATTERNS) {
    let m: RegExpExecArray | null;
    while ((m = p.regex.exec(text)) !== null) {
      found.push({
        kind: p.kind,
        text: m[0],
        range: { start: m.index, end: m.index + m[0].length },
      });
    }
  }

  for (const term of [...USER_DICT, ...DOMAIN_ENTITIES]) {
    let offset = 0;
    while (offset < text.length) {
      const idx = text.indexOf(term, offset);
      if (idx < 0) break;
      found.push({
        kind: USER_DICT.includes(term) ? 'USER_DICT' : 'ENTITY',
        text: term,
        range: { start: idx, end: idx + term.length },
      });
      offset = idx + term.length;
    }
  }

  found.sort((a, b) => a.range.start - b.range.start || a.range.end - b.range.end);
  const merged: Array<Omit<ProtectedSpan, 'placeholder'>> = [];
  for (const item of found) {
    const last = merged[merged.length - 1];
    if (!last || !overlap(last.range, item.range)) {
      merged.push(item);
      continue;
    }
    last.range = { start: Math.min(last.range.start, item.range.start), end: Math.max(last.range.end, item.range.end) };
    last.text = text.slice(last.range.start, last.range.end);
  }

  const protectedSpans: ProtectedSpan[] = merged.map((m, i) => ({ ...m, placeholder: `__${m.kind}_${i}__` }));

  let masked = '';
  let cursor = 0;
  for (const span of protectedSpans) {
    masked += text.slice(cursor, span.range.start);
    masked += span.placeholder;
    cursor = span.range.end;
  }
  masked += text.slice(cursor);

  return { protectedSpans, maskedText: masked };
}

export function isProtected(range: TextRange, protectedSpans: ProtectedSpan[]): boolean {
  return protectedSpans.some((p) => overlap(range, p.range));
}
