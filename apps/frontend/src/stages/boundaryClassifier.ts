import type { Profile, TextEdit } from '../pipeline/types';
import type { BoundaryCandidate } from './spacingProposer';

export function classifySpacingBoundaries(
  text: string,
  candidates: BoundaryCandidate[],
  profile: Profile
): BoundaryCandidate[] {
  return candidates
    .map((c) => {
      let score = c.score + (profile === 'NORMAL' ? 0.04 : 0);
      const left = text.slice(Math.max(0, c.index - 3), c.index);
      const right = text.slice(c.index, Math.min(text.length, c.index + 3));
      if (/오늘|산책/.test(left)) score += 0.05;
      if (/날씨|너무|좋은|가도|돼요/.test(right)) score += 0.05;
      return { ...c, score: Math.min(0.99, score) };
    })
    .filter((c) => c.score >= 0.9);
}

export function spacingToEdits(candidates: BoundaryCandidate[]): TextEdit[] {
  return candidates.map((c) => ({
    stage: 'SPACING',
    range: { start: c.index, end: c.action === 'INSERT_SPACE' ? c.index : c.index + 1 },
    sourceText: c.action === 'INSERT_SPACE' ? '' : ' ',
    replacement: c.action === 'INSERT_SPACE' ? ' ' : '',
    editType: c.action === 'INSERT_SPACE' ? 'SPACE_INSERT' : 'SPACE_DELETE',
    confidence: c.score,
    autoApplicable: true,
    reasonTag: 'spacing_boundary',
  }));
}
