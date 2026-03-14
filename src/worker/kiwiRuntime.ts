/// <reference lib="webworker" />

import { KiwiBuilder, Match, type Kiwi, type SentenceJoinMorph, type TokenInfo } from 'kiwi-nlp';
import kiwiWasmPath from 'kiwi-nlp/dist/kiwi-wasm.wasm?url';

export interface KiwiMorph {
  form: string;
  tag: string;
  position: number;
  length: number;
  wordPosition: number;
}

export interface KiwiCanonicalToken {
  surface: string;
  range: { start: number; end: number };
  morphs: KiwiMorph[];
  headIndex: number;
  lemmaKey: string | null;
  pos: string | null;
  slotTags: string[];
}

export interface KiwiCanonicalAnalysis {
  ready: boolean;
  version?: string;
  error?: string;
  tokensByRange: Map<string, KiwiCanonicalToken>;
}

interface KiwiRuntimeState {
  builder: KiwiBuilder;
  kiwi: Kiwi;
  version: string;
}

const MODEL_FILES = [
  'combiningRule.txt',
  'cong.mdl',
  'default.dict',
  'dialect.dict',
  'extract.mdl',
  'multi.dict',
  'nounchr.mdl',
  'sj.morph',
  'typo.dict',
] as const;

const PREDICATE_TAGS = new Set(['VV', 'VA', 'VX', 'VCP', 'VCN']);

let runtimePromise: Promise<KiwiRuntimeState | null> | null = null;
let runtimeState: KiwiRuntimeState | null = null;
let runtimeError: string | null = null;

function rangeKey(start: number, end: number): string {
  return `${start}:${end}`;
}

function whitespaceTokens(text: string): Array<{ token: string; start: number; end: number }> {
  const out: Array<{ token: string; start: number; end: number }> = [];
  const re = /\S+/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    out.push({ token: match[0], start: match.index, end: match.index + match[0].length });
  }
  return out;
}

function isPredicateTag(tag: string): boolean {
  if (PREDICATE_TAGS.has(tag)) return true;
  return tag.startsWith('VV') || tag.startsWith('VA') || tag.startsWith('VX') || tag.startsWith('VCP') || tag.startsWith('VCN');
}

function lemmaKeyFor(form: string, tag: string): string {
  return isPredicateTag(tag) ? (form.endsWith('다') ? form : `${form}다`) : form;
}

function modelBaseUrl(): string {
  return new URL(`${import.meta.env.BASE_URL}assets/kiwi/model/`, self.location.origin).toString();
}

function modelFilesMap(): Record<string, string> {
  const base = modelBaseUrl();
  return Object.fromEntries(MODEL_FILES.map((file) => [file, `${base}${file}`]));
}

async function createRuntime(): Promise<KiwiRuntimeState | null> {
  try {
    const builder = await KiwiBuilder.create(kiwiWasmPath);
    const kiwi = await builder.build({
      modelFiles: modelFilesMap(),
      modelType: 'cong',
      loadDefaultDict: true,
      loadMultiDict: true,
      loadTypoDict: true,
      typos: 'basic',
      typoCostThreshold: 2.5,
      integrateAllomorph: true,
    });

    const state = {
      builder,
      kiwi,
      version: builder.version(),
    };
    runtimeState = state;
    runtimeError = null;
    return state;
  } catch (error) {
    runtimeError = error instanceof Error ? error.message : 'Kiwi runtime initialization failed';
    runtimeState = null;
    return null;
  }
}

export async function ensureKiwiRuntime(): Promise<KiwiRuntimeState | null> {
  if (runtimeState) return runtimeState;
  if (!runtimePromise) {
    runtimePromise = createRuntime().finally(() => {
      runtimePromise = null;
    });
  }
  return runtimePromise;
}

export function getKiwiRuntimeStatus(): { ready: boolean; version?: string; error?: string } {
  if (runtimeState) {
    return { ready: true, version: runtimeState.version };
  }
  return { ready: false, error: runtimeError ?? undefined };
}

function buildCanonicalToken(
  token: { token: string; start: number; end: number },
  morphs: KiwiMorph[]
): KiwiCanonicalToken {
  const headIndex = morphs.findIndex((morph) => isPredicateTag(morph.tag));
  const head = headIndex >= 0 ? morphs[headIndex] : null;
  return {
    surface: token.token,
    range: { start: token.start, end: token.end },
    morphs,
    headIndex,
    lemmaKey: head ? lemmaKeyFor(head.form, head.tag) : null,
    pos: head?.tag ?? null,
    slotTags: headIndex >= 0 ? morphs.slice(headIndex + 1).map((morph) => morph.tag) : [],
  };
}

export async function analyzeCanonicalTokens(text: string): Promise<KiwiCanonicalAnalysis> {
  const runtime = await ensureKiwiRuntime();
  if (!runtime) {
    return {
      ready: false,
      error: runtimeError ?? 'Kiwi runtime unavailable',
      tokensByRange: new Map(),
    };
  }

  try {
    const tokens = runtime.kiwi.tokenize(text, Match.allWithNormalizing);
    const spacedTokens = whitespaceTokens(text);
    const byRange = new Map<string, KiwiCanonicalToken>();

    for (const token of spacedTokens) {
      const morphs = tokens
        .filter((morph) => morph.position >= token.start && morph.position < token.end)
        .map(
          (morph): KiwiMorph => ({
            form: morph.str,
            tag: morph.tag,
            position: morph.position,
            length: morph.length,
            wordPosition: morph.wordPosition,
          })
        );
      if (!morphs.length) continue;
      byRange.set(rangeKey(token.start, token.end), buildCanonicalToken(token, morphs));
    }

    return {
      ready: true,
      version: runtime.version,
      tokensByRange: byRange,
    };
  } catch (error) {
    return {
      ready: false,
      version: runtime.version,
      error: error instanceof Error ? error.message : 'Kiwi canonical analysis failed',
      tokensByRange: new Map(),
    };
  }
}

export function lookupCanonicalToken(
  analysis: KiwiCanonicalAnalysis | undefined,
  range: { start: number; end: number }
): KiwiCanonicalToken | undefined {
  if (!analysis?.ready) return undefined;
  return analysis.tokensByRange.get(rangeKey(range.start, range.end));
}

export async function reinflectCanonicalToken(
  canonicalToken: KiwiCanonicalToken,
  replacementLemmaKey: string
): Promise<string | null> {
  const runtime = await ensureKiwiRuntime();
  if (!runtime) return null;
  if (!canonicalToken.morphs.length) return null;

  const replacementStem = replacementLemmaKey.endsWith('다')
    ? replacementLemmaKey.slice(0, -1)
    : replacementLemmaKey;

  const morphs: SentenceJoinMorph[] = canonicalToken.morphs.map((morph, index) => ({
    form: index === canonicalToken.headIndex ? replacementStem : morph.form,
    tag: morph.tag,
  }));

  try {
    const joined = runtime.kiwi.joinSent(morphs, true, false);
    const normalized = joined?.str?.trim();
    return normalized && normalized !== canonicalToken.surface ? normalized : null;
  } catch {
    return null;
  }
}
