import type { Candidate } from '../pipeline/types';
import type { BrowserRuntimeAssets } from '../worker/runtimeAssets';
import type { KiwiCanonicalAnalysis, KiwiCanonicalToken } from '../worker/kiwiRuntime';
import { lookupCanonicalToken } from '../worker/kiwiRuntime';

function slotSignature(canonical?: KiwiCanonicalToken): string | null {
  if (!canonical?.slotTags?.length) return null;
  return canonical.slotTags.join('+');
}

function hasContextHint(text: string, contextHints?: string[]): boolean {
  if (!contextHints?.length) return false;
  return contextHints.some((term) => term && text.includes(term));
}

export function verifyCandidates(
  text: string,
  candidates: Candidate[],
  assets: BrowserRuntimeAssets,
  canonicalAnalysis?: KiwiCanonicalAnalysis
): Candidate[] {
  return candidates
    .map((candidate) => {
      if (candidate.source === 'ORIGINAL') {
        return { ...candidate, verifyScore: 0.25 };
      }

      const canonical = lookupCanonicalToken(canonicalAnalysis, candidate.span);
      const slotKey = slotSignature(canonical);
      let score = 0.35 + candidate.generatorScore * 0.35;

      if (candidate.typoDistance != null) {
        if (candidate.typoDistance <= 1) score += 0.18;
        else if (candidate.typoDistance >= 4) score -= 0.25;
      }

      if (candidate.posHint && canonical?.pos) {
        if (candidate.posHint === canonical.pos) score += 0.12;
        else score -= 0.2;
      }

      if (candidate.slotHints?.length) {
        if (slotKey && candidate.slotHints.includes(slotKey)) score += 0.14;
        else if (slotKey) score -= 0.18;
      }

      if (candidate.familyLemma && canonical?.lemmaKey) {
        if (candidate.familyLemma === canonical.lemmaKey) score -= 0.08;
        else score += 0.06;
      }

      if (hasContextHint(text, candidate.contextHints)) {
        score += 0.12;
      } else if (candidate.source === 'AUTO_FAMILY_SEED') {
        score -= 0.08;
      }

      if (candidate.source === 'GEN_BACKOFF') {
        score -= 0.1;
      }

      return {
        ...candidate,
        verifyScore: Math.max(0.01, Math.min(0.99, score)),
      };
    })
    .filter((candidate) => candidate.source === 'ORIGINAL' || (candidate.verifyScore ?? 0) >= 0.36)
    .sort((a, b) => (b.verifyScore ?? 0) - (a.verifyScore ?? 0) || b.generatorScore - a.generatorScore);
}
