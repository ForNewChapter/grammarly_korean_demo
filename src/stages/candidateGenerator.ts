import type { Candidate } from '../pipeline/types';
import type { TaggedToken } from './editTagger';

const CONFUSION_SET: Record<string, Array<{ replacement: string; source: Candidate['source']; score: number }>> = {
  낫다: [
    { replacement: '낳다', source: 'CONFUSION_SET', score: 0.94 },
    { replacement: '놓다', source: 'MORPH', score: 0.41 },
  ],
  낫아: [{ replacement: '낳아', source: 'MORPH', score: 0.9 }],
  안되: [{ replacement: '안 돼', source: 'CONFUSION_SET', score: 0.88 }],
};

function stemMorphCandidates(token: string): Array<{ replacement: string; source: Candidate['source']; score: number }> {
  const out: Array<{ replacement: string; source: Candidate['source']; score: number }> = [];
  if (/^낫(아|어|았|었|으면|으니|지)/.test(token)) {
    out.push({ replacement: token.replace(/^낫/, '낳'), source: 'MORPH', score: 0.87 });
  }
  if (token.includes('됬')) {
    out.push({ replacement: token.replaceAll('됬', '됐'), source: 'JAMO', score: 0.82 });
  }
  return out;
}

export function generateCandidates(text: string, tags: TaggedToken[]): Candidate[] {
  const out: Candidate[] = [];
  for (const t of tags) {
    if (t.label !== 'OPEN_REPLACE') continue;

    const base = CONFUSION_SET[t.token] ?? [];
    const morph = stemMorphCandidates(t.token);
    const all = [...base, ...morph];

    const dedup = new Map<string, { replacement: string; source: Candidate['source']; score: number }>();
    for (const c of all) {
      const existing = dedup.get(c.replacement);
      if (!existing || c.score > existing.score) dedup.set(c.replacement, c);
    }

    for (const c of dedup.values()) {
      out.push({
        span: t.range,
        original: t.token,
        replacement: c.replacement,
        source: c.source,
        generatorScore: c.score,
      });
    }

    out.push({
      span: t.range,
      original: t.token,
      replacement: t.token,
      source: 'ORIGINAL',
      generatorScore: 0.2,
    });
  }
  return out;
}
