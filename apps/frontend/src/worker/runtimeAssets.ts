import { RULE_ASSETS } from '../cache/assetManifest';

export interface SurfaceFixRule {
  from: string;
  to: string;
  reasonTag: string;
  confidence: number;
}

export interface PhraseRule {
  from: string;
  to: string;
  reasonTag: string;
  confidence: number;
}

export interface RuntimeCandidateSeed {
  replacement: string;
  source?: string;
  generatorScore?: number;
}

export interface RuntimeFamilySeed extends RuntimeCandidateSeed {
  familyLemma?: string;
  pos?: string;
  suffixSlots?: string[];
  contextHints?: string[];
}

export interface InflectionRule {
  sourceSuffix: string;
  targetSuffix: string;
  generatorScore: number;
}

export interface BrowserRuntimeAssets {
  ready: boolean;
  surfaceFixRules: SurfaceFixRule[];
  surfaceFixMap: Map<string, SurfaceFixRule>;
  phraseRules: PhraseRule[];
  phraseSourceTerms: Set<string>;
  typedSurfaceMap: Map<string, RuntimeCandidateSeed[]>;
  familyLemmaMap: Map<string, RuntimeFamilySeed[]>;
  familySurfaceMap: Map<string, RuntimeFamilySeed[]>;
  inflectionRules: InflectionRule[];
  errors: string[];
}

const EMPTY: BrowserRuntimeAssets = {
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

let cachedAssets: BrowserRuntimeAssets | null = null;

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function mergeSurfaceRules(primary: SurfaceFixRule[], secondary: SurfaceFixRule[]): SurfaceFixRule[] {
  const merged = new Map<string, SurfaceFixRule>();
  for (const item of [...primary, ...secondary]) {
    if (!item?.from || !item?.to) continue;
    const key = `${item.from}\u0000${item.to}`;
    const prev = merged.get(key);
    if (!prev || item.confidence >= prev.confidence) {
      merged.set(key, item);
    }
  }
  return [...merged.values()].sort((a, b) => b.from.length - a.from.length || b.confidence - a.confidence);
}

function pushSeed(map: Map<string, RuntimeCandidateSeed[]>, key: string, row: RuntimeCandidateSeed): void {
  if (!key || !row.replacement) return;
  const list = map.get(key) ?? [];
  const existing = list.find((item) => item.replacement === row.replacement);
  if (!existing || (row.generatorScore ?? 0) > (existing.generatorScore ?? 0)) {
    const next = list.filter((item) => item.replacement !== row.replacement);
    next.push(row);
    next.sort((a, b) => (b.generatorScore ?? 0) - (a.generatorScore ?? 0));
    map.set(key, next.slice(0, 8));
  }
}

function pushFamilySeed(map: Map<string, RuntimeFamilySeed[]>, key: string, row: RuntimeFamilySeed): void {
  if (!key || !row.replacement) return;
  const list = map.get(key) ?? [];
  const existing = list.find((item) => item.replacement === row.replacement);
  if (!existing || (row.generatorScore ?? 0) > (existing.generatorScore ?? 0)) {
    const next = list.filter((item) => item.replacement !== row.replacement);
    next.push({
      replacement: row.replacement,
      source: row.source,
      generatorScore: row.generatorScore,
      familyLemma: row.familyLemma,
      pos: row.pos,
      suffixSlots: row.suffixSlots ?? [],
      contextHints: row.contextHints ?? [],
    });
    next.sort((a, b) => (b.generatorScore ?? 0) - (a.generatorScore ?? 0));
    map.set(key, next.slice(0, 8));
  }
}

function normalizeTypedSurfaceMap(payload: unknown): Map<string, RuntimeCandidateSeed[]> {
  const out = new Map<string, RuntimeCandidateSeed[]>();
  const surface = (payload as { surface?: Record<string, any[]> })?.surface ?? {};
  for (const [key, rows] of Object.entries(surface)) {
    for (const row of asArray<any>(rows)) {
      pushSeed(out, key, {
        replacement: String(row.replacement ?? ''),
        source: String(row.source ?? 'DATA_CONFUSION'),
        generatorScore: Number(row.generatorScore ?? 0.6),
      });
    }
  }
  return out;
}

function normalizeFamilySeedMaps(payload: unknown): {
  lemmaMap: Map<string, RuntimeFamilySeed[]>;
  surfaceMap: Map<string, RuntimeFamilySeed[]>;
} {
  const lemmaMap = new Map<string, RuntimeFamilySeed[]>();
  const surfaceMap = new Map<string, RuntimeFamilySeed[]>();
  const lemmas = (payload as { lemmas?: Record<string, any[]> })?.lemmas ?? {};
  for (const [lemma, rows] of Object.entries(lemmas)) {
    for (const row of asArray<any>(rows)) {
      const normalizedRow: RuntimeFamilySeed = {
        replacement: String(row.replacement ?? ''),
        source: String(row.source ?? 'FAMILY_SEED'),
        generatorScore: Number(row.generatorScore ?? 0.78),
        familyLemma: String(row.replacement ?? ''),
        pos: row.pos ? String(row.pos) : undefined,
        suffixSlots: asArray<string>(row.suffixSlots).map((slot) => String(slot)),
        contextHints: asArray<any>(row.contextHints).map((hint) => String(hint?.term ?? '')).filter(Boolean),
      };
      pushFamilySeed(lemmaMap, lemma, normalizedRow);
      for (const sample of asArray<any>(row.surfaceExamples)) {
        if (sample?.from && sample?.to) {
          pushFamilySeed(surfaceMap, String(sample.from), {
            replacement: String(sample.to),
            source: 'FAMILY_SURFACE_EXAMPLE',
            generatorScore: Math.max(0.72, Number(row.generatorScore ?? 0.78)),
            familyLemma: String(row.replacement ?? ''),
            pos: row.pos ? String(row.pos) : undefined,
            suffixSlots: asArray<string>(row.suffixSlots).map((slot) => String(slot)),
            contextHints: asArray<any>(row.contextHints).map((hint) => String(hint?.term ?? '')).filter(Boolean),
          });
        }
      }
    }
  }
  return { lemmaMap, surfaceMap };
}

function normalizeInflectionRules(payload: unknown): InflectionRule[] {
  return asArray<any>((payload as { rules?: any[] })?.rules)
    .map((item: any) => ({
      sourceSuffix: String(item.sourceSuffix ?? ''),
      targetSuffix: String(item.targetSuffix ?? ''),
      generatorScore: Number(item.generatorScore ?? 0.64),
    }))
    .filter((item: InflectionRule) => item.sourceSuffix && item.targetSuffix && item.sourceSuffix !== item.targetSuffix)
    .sort(
      (a: InflectionRule, b: InflectionRule) =>
        b.sourceSuffix.length - a.sourceSuffix.length || b.generatorScore - a.generatorScore
    );
}

async function fetchJson(path: string): Promise<unknown> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export async function loadRuntimeAssets(): Promise<BrowserRuntimeAssets> {
  if (cachedAssets) return cachedAssets;

  const errors: string[] = [];
  let commonRules: SurfaceFixRule[] = [];
  let highPrecisionRules: SurfaceFixRule[] = [];
  let phraseRules: PhraseRule[] = [];
  let typedSurfaceMap = new Map<string, RuntimeCandidateSeed[]>();
  let familyLemmaMap = new Map<string, RuntimeFamilySeed[]>();
  let familySurfaceMap = new Map<string, RuntimeFamilySeed[]>();
  let inflectionRules: InflectionRule[] = [];

  const results = await Promise.all(
    RULE_ASSETS.map(async (path) => {
      try {
        return { path, payload: await fetchJson(path) };
      } catch (error) {
        errors.push(error instanceof Error ? error.message : `${path} load failed`);
        return { path, payload: null };
      }
    })
  );

  for (const { path, payload } of results) {
    if (!payload) continue;
    if (path.endsWith('/common_misspellings.json')) {
      commonRules = asArray<SurfaceFixRule>(payload).map((item) => ({
        from: String(item.from),
        to: String(item.to),
        reasonTag: String(item.reasonTag ?? 'common_misspelling'),
        confidence: Number(item.confidence ?? 0.95),
      }));
      continue;
    }
    if (path.endsWith('/high_precision_surface_fixes.json')) {
      highPrecisionRules = asArray<any>((payload as { items?: any[] }).items).map((item) => ({
        from: String(item.from),
        to: String(item.to),
        reasonTag: String(item.reasonTag ?? 'high_precision_surface_fix'),
        confidence: Number(item.confidence ?? 0.95),
      }));
      continue;
    }
    if (path.endsWith('/phrase_memory.json')) {
      phraseRules = asArray<any>((payload as { items?: any[] }).items).map((item) => ({
        from: String(item.from),
        to: String(item.to),
        reasonTag: String(item.reasonTag ?? 'phrase_memory'),
        confidence: Number(item.confidence ?? 0.96),
      }));
      continue;
    }
    if (path.endsWith('/typed_confusion_graph.json')) {
      typedSurfaceMap = normalizeTypedSurfaceMap(payload);
      continue;
    }
    if (path.endsWith('/predicate_family_seeds.json')) {
      const familyMaps = normalizeFamilySeedMaps(payload);
      familyLemmaMap = familyMaps.lemmaMap;
      familySurfaceMap = familyMaps.surfaceMap;
      continue;
    }
    if (path.endsWith('/inflection_recovery_rules.json')) {
      inflectionRules = normalizeInflectionRules(payload);
    }
  }

  const surfaceFixRules = mergeSurfaceRules(highPrecisionRules, commonRules);
  const surfaceFixMap = new Map(surfaceFixRules.map((item) => [item.from, item]));
  const phraseSourceTerms = new Set(phraseRules.map((item) => item.from));

  cachedAssets = {
    ready:
      surfaceFixRules.length > 0 &&
      phraseRules.length > 0 &&
      typedSurfaceMap.size > 0 &&
      familyLemmaMap.size > 0 &&
      inflectionRules.length > 0,
    surfaceFixRules,
    surfaceFixMap,
    phraseRules,
    phraseSourceTerms,
    typedSurfaceMap,
    familyLemmaMap,
    familySurfaceMap,
    inflectionRules,
    errors,
  };

  return cachedAssets;
}

export function getRuntimeAssets(): BrowserRuntimeAssets {
  return cachedAssets ?? EMPTY;
}
