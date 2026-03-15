import type { Candidate } from '../pipeline/types';
import type { TaggedToken } from './editTagger';
import type { BrowserRuntimeAssets, RuntimeCandidateSeed, RuntimeFamilySeed, InflectionRule } from '../worker/runtimeAssets';
import type { KiwiCanonicalAnalysis } from '../worker/kiwiRuntime';
import { lookupCanonicalToken, reinflectCanonicalToken } from '../worker/kiwiRuntime';
import type { KiwiCanonicalToken } from '../worker/kiwiRuntime';

function charEditDistance(a: string, b: string): number {
  const dp = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
  for (let i = 0; i <= a.length; i += 1) dp[i][0] = i;
  for (let j = 0; j <= b.length; j += 1) dp[0][j] = j;
  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[a.length][b.length];
}

function pushCandidate(
  dedup: Map<string, Candidate>,
  token: TaggedToken,
  replacement: string,
  source: Candidate['source'],
  generatorScore: number,
  meta?: Partial<Candidate>
): void {
  if (!replacement || replacement === token.token) return;
  const distance = charEditDistance(token.token, replacement);
  if (distance > Math.max(3, Math.ceil(token.token.length * 0.5))) return;
  const existing = dedup.get(replacement);
  if (!existing || generatorScore > existing.generatorScore) {
    dedup.set(replacement, {
      span: token.range,
      original: token.token,
      replacement,
      source,
      generatorScore,
      typoDistance: distance,
      ...meta,
    });
  }
}

function applySeedRows(
  dedup: Map<string, Candidate>,
  token: TaggedToken,
  rows: RuntimeCandidateSeed[] | undefined,
  sourceFallback: Candidate['source']
): void {
  for (const row of rows ?? []) {
    pushCandidate(
      dedup,
      token,
      row.replacement,
      (row.source as Candidate['source']) ?? sourceFallback,
      row.generatorScore ?? 0.5
    );
  }
}

function slotSignature(canonical?: KiwiCanonicalToken): string | null {
  if (!canonical?.slotTags?.length) return null;
  return canonical.slotTags.join('+');
}

async function applyCanonicalFamilyRows(
  dedup: Map<string, Candidate>,
  token: TaggedToken,
  analysis: KiwiCanonicalAnalysis | undefined,
  assets: BrowserRuntimeAssets
): Promise<void> {
  const canonical = lookupCanonicalToken(analysis, token.range);
  if (!canonical?.lemmaKey) return;

  const rows = assets.familyLemmaMap.get(canonical.lemmaKey) ?? [];
  for (const row of rows) {
    const reinflected = await reinflectCanonicalToken(canonical, row.replacement);
    const meta: Partial<Candidate> = {
      familyLemma: row.familyLemma ?? row.replacement,
      posHint: row.pos,
      slotHints: row.suffixSlots ?? [],
      contextHints: row.contextHints ?? [],
    };
    if (reinflected) {
      pushCandidate(
        dedup,
        token,
        reinflected,
        (row.source as Candidate['source']) ?? 'FAMILY_SEED',
        row.generatorScore ?? 0.78,
        meta
      );
      continue;
    }

    pushCandidate(
      dedup,
      token,
      row.replacement,
      (row.source as Candidate['source']) ?? 'FAMILY_SEED',
      row.generatorScore ?? 0.72,
      meta
    );
  }
}

function applyFamilySurfaceRows(
  dedup: Map<string, Candidate>,
  token: TaggedToken,
  rows: RuntimeFamilySeed[] | undefined
): void {
  for (const row of rows ?? []) {
    pushCandidate(
      dedup,
      token,
      row.replacement,
      (row.source as Candidate['source']) ?? 'FAMILY_SURFACE_EXAMPLE',
      row.generatorScore ?? 0.72,
      {
        familyLemma: row.familyLemma ?? row.replacement,
        posHint: row.pos,
        slotHints: row.suffixSlots ?? [],
        contextHints: row.contextHints ?? [],
      }
    );
  }
}

function applyInflectionRules(
  dedup: Map<string, Candidate>,
  token: TaggedToken,
  rules: InflectionRule[]
): void {
  for (const rule of rules) {
    if (token.token.length <= rule.sourceSuffix.length) continue;
    if (!token.token.endsWith(rule.sourceSuffix)) continue;
    const stem = token.token.slice(0, token.token.length - rule.sourceSuffix.length);
    const replacement = `${stem}${rule.targetSuffix}`;
    pushCandidate(dedup, token, replacement, 'MORPH', rule.generatorScore);
  }
}

function applyJamoFallback(
  dedup: Map<string, Candidate>,
  token: TaggedToken,
  assets: BrowserRuntimeAssets
): void {
  if (token.token.length < 2) return;
  for (const [surface, rows] of assets.typedSurfaceMap.entries()) {
    if (surface === token.token) continue;
    if (Math.abs(surface.length - token.token.length) > 1) continue;
    if (charEditDistance(surface, token.token) > 1) continue;
    applySeedRows(dedup, token, rows, 'GEN_BACKOFF');
    if (dedup.size >= 6) return;
  }
}

export async function generateCandidates(
  text: string,
  tags: TaggedToken[],
  assets?: BrowserRuntimeAssets,
  canonicalAnalysis?: KiwiCanonicalAnalysis
): Promise<Candidate[]> {
  const out: Candidate[] = [];
  const runtimeAssets: BrowserRuntimeAssets = assets ?? {
    ready: false,
    surfaceFixRules: [],
    surfaceFixMap: new Map(),
    phraseRules: [],
    phraseSourceTerms: new Set(),
    typedSurfaceMap: new Map(),
    familyLemmaMap: new Map(),
    familySurfaceMap: new Map(),
    inflectionRules: [],
    errors: [],
  };

  for (const t of tags) {
    if (t.label !== 'OPEN_REPLACE') continue;

    const dedup = new Map<string, Candidate>();

    const directSurfaceRule = runtimeAssets.surfaceFixMap.get(t.token);
    if (directSurfaceRule) {
      pushCandidate(dedup, t, directSurfaceRule.to, 'CONFUSION_SET', directSurfaceRule.confidence);
    }

    applySeedRows(dedup, t, runtimeAssets.typedSurfaceMap.get(t.token), 'CONFUSION_SET');
    applyFamilySurfaceRows(dedup, t, runtimeAssets.familySurfaceMap.get(t.token));
    await applyCanonicalFamilyRows(dedup, t, canonicalAnalysis, runtimeAssets);
    applyInflectionRules(dedup, t, runtimeAssets.inflectionRules);
    applyJamoFallback(dedup, t, runtimeAssets);

    out.push(...[...dedup.values()].sort((a, b) => b.generatorScore - a.generatorScore).slice(0, 6));
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
