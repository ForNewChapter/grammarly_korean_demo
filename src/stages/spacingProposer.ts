import type { Profile, ProtectedSpan } from '../pipeline/types';
import { isProtected } from './protectedSpan';

export interface BoundaryCandidate {
  index: number;
  action: 'INSERT_SPACE' | 'DELETE_SPACE';
  score: number;
}

const HINTS = ['오늘', '날씨가', '너무', '좋은데', '산책', '가도', '할', '수'];

export function proposeSpacing(text: string, profile: Profile, protectedSpans: ProtectedSpan[]): BoundaryCandidate[] {
  const out: BoundaryCandidate[] = [];
  for (let i = 1; i < text.length; i += 1) {
    const l = text[i - 1];
    const r = text[i];
    if (!/[가-힣]/.test(l) || !/[가-힣]/.test(r)) continue;
    if (l === ' ' || r === ' ') continue;
    if (isProtected({ start: i - 1, end: i + 1 }, protectedSpans)) continue;

    const window = text.slice(Math.max(0, i - 4), Math.min(text.length, i + 4));
    let score = 0.42;
    for (const h of HINTS) {
      if (window.includes(h.slice(0, 2)) || window.includes(h.slice(-2))) score += 0.1;
    }
    if (/가도|는데|어요|했다|하면/.test(window)) score += 0.15;
    if (profile === 'NOISY') score += 0.03;
    if (score >= 0.8) out.push({ index: i, action: 'INSERT_SPACE', score: Math.min(score, 0.99) });
  }
  return out;
}
