import type { Candidate, Profile } from '../pipeline/types';

export function rerankCandidates(
  text: string,
  profile: Profile,
  candidates: Candidate[],
  taggerConfidence: Record<string, number>
): Candidate[] {
  return candidates
    .map((c) => {
      let rerankScore = 0.28 + (c.verifyScore ?? 0.3) * 0.18;
      const sentence = `${text.slice(0, c.span.start)}${c.replacement}${text.slice(c.span.end)}`;
      if (/아기/.test(sentence) && /낳/.test(sentence)) rerankScore += 0.55;
      if (/놓/.test(sentence)) rerankScore -= 0.2;
      if (c.replacement === c.original) rerankScore -= 0.08;
      if (profile === 'CHAT') rerankScore -= 0.04;

      const conf = taggerConfidence[`${c.span.start}:${c.span.end}`] ?? 0.8;
      const finalScore = Math.min(
        0.99,
        Math.max(0.01, rerankScore) + 0.12 * c.generatorScore + 0.12 * (c.verifyScore ?? 0.3) + 0.08 * conf
      );
      return {
        ...c,
        rerankScore: Math.max(0.01, Math.min(0.99, rerankScore)),
        finalScore,
      };
    })
    .sort((a, b) => (b.finalScore ?? 0) - (a.finalScore ?? 0));
}
