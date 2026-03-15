/// <reference lib="webworker" />

import type { Candidate, GuardrailDecision, PipelineResult, Profile, StageTrace, TextEdit } from '../pipeline/types';
import type { WorkerRequest, WorkerResponse } from './protocol';
import { ModelManager } from './modelManager';
import { extractRange } from '../stages/rangeExtractor';
import { detectProtectedSpans } from '../stages/protectedSpan';
import { normalizeText } from '../stages/normalizer';
import { classifyProfile } from '../stages/profileClassifier';
import { runRuleCorrector } from '../stages/ruleCorrector';
import { runPhrasePreNormalizer } from '../stages/phrasePreNormalizer';
import { proposeSpacing } from '../stages/spacingProposer';
import { classifySpacingBoundaries, spacingToEdits } from '../stages/boundaryClassifier';
import { runEditTagger } from '../stages/editTagger';
import type { TaggedToken } from '../stages/editTagger';
import { generateCandidates } from '../stages/candidateGenerator';
import { verifyCandidates } from '../stages/candidateVerifier';
import { rerankCandidates } from '../stages/reranker';
import { guardrailDecision, policyForEdit } from '../stages/guardrail';
import { runPolicyEngine } from '../stages/policyEngine';
import { applyEdits } from '../pipeline/patchApplier';
import { makeTrace } from '../pipeline/trace';
import {
  argmax,
  blend,
  makeBoundaryFeatures,
  makeRerankFeatures,
  makeTextFeatures,
  makeTokenFeatures,
  runNumericModel,
  scoreFromVector,
  softmaxConfidence,
} from './onnxHelpers';
import { loadRuntimeAssets } from './runtimeAssets';
import { analyzeCanonicalTokens, ensureKiwiRuntime, getKiwiRuntimeStatus } from './kiwiRuntime';
import type { KiwiCanonicalAnalysis } from './kiwiRuntime';
import type { ProtectedSpan } from '../pipeline/types';
import { isProtected } from '../stages/protectedSpan';
import {
  getBrowserEditTaggerStatus,
  predictBrowserEditTagger,
  warmupBrowserEditTagger,
} from './browserEditTaggerModel';

const modelManager = new ModelManager();
const PROFILE_LABELS: Profile[] = ['NORMAL', 'CHAT', 'NOISY', 'MIXED', 'QUERY'];

function profileCode(profile: Profile): number {
  return PROFILE_LABELS.indexOf(profile) / PROFILE_LABELS.length;
}

function now(): number {
  return performance.now();
}

function groupBestBySpan(candidates: Candidate[]): Candidate[] {
  const map = new Map<string, Candidate>();
  for (const c of candidates) {
    const key = `${c.span.start}:${c.span.end}`;
    const existing = map.get(key);
    if (!existing || (c.finalScore ?? 0) > (existing.finalScore ?? 0)) {
      map.set(key, c);
    }
  }
  return [...map.values()];
}

function groupCandidatesForTrace(candidates: Candidate[]): Array<{
  original: string;
  items: Array<{ replacement: string; source: string; finalScore?: number; verifyScore?: number }>;
}> {
  const grouped = new Map<string, {
    original: string;
    items: Array<{ replacement: string; source: string; finalScore?: number; verifyScore?: number }>;
  }>();

  for (const candidate of candidates) {
    const key = `${candidate.span.start}:${candidate.span.end}`;
    const current = grouped.get(key) ?? { original: candidate.original, items: [] };
    current.items.push({
      replacement: candidate.replacement,
      source: candidate.source,
      finalScore: candidate.finalScore,
      verifyScore: candidate.verifyScore,
    });
    grouped.set(key, current);
  }

  return [...grouped.values()].map((group) => ({
    ...group,
    items: group.items.sort((a, b) => (b.finalScore ?? 0) - (a.finalScore ?? 0)),
  }));
}

function groupVerifierForTrace(generated: Candidate[], verified: Candidate[]): Array<{
  original: string;
  candidateVerifier: { inputCount: number; keptCount: number; keptSources: string[] };
}> {
  const inputCounts = new Map<string, { original: string; count: number }>();
  for (const candidate of generated) {
    const key = `${candidate.span.start}:${candidate.span.end}`;
    const current = inputCounts.get(key) ?? { original: candidate.original, count: 0 };
    current.count += 1;
    inputCounts.set(key, current);
  }

  const kept = new Map<string, { count: number; sources: Set<string> }>();
  for (const candidate of verified) {
    const key = `${candidate.span.start}:${candidate.span.end}`;
    const current = kept.get(key) ?? { count: 0, sources: new Set<string>() };
    current.count += 1;
    current.sources.add(candidate.source);
    kept.set(key, current);
  }

  return [...inputCounts.entries()].map(([key, input]) => ({
    original: input.original,
    candidateVerifier: {
      inputCount: input.count,
      keptCount: kept.get(key)?.count ?? 0,
      keptSources: [...(kept.get(key)?.sources ?? new Set<string>())],
    },
  }));
}

function groupRerankerForTrace(candidates: Candidate[]): Array<{
  original: string;
  best?: { replacement: string; finalScore: number };
}> {
  return groupBestBySpan(candidates).map((candidate) => ({
    original: candidate.original,
    best: {
      replacement: candidate.replacement,
      finalScore: candidate.finalScore ?? 0,
    },
  }));
}

function kiwiSpacingFallback(
  text: string,
  analysis: KiwiCanonicalAnalysis,
  protectedSpans: ProtectedSpan[]
): Array<{ index: number; action: 'INSERT_SPACE'; score: number }> {
  if (!analysis.ready || analysis.tokens.length < 2) return [];
  const out: Array<{ index: number; action: 'INSERT_SPACE'; score: number }> = [];
  const ordered = [...analysis.tokens].sort((a, b) => a.range.start - b.range.start);
  for (let i = 1; i < ordered.length; i += 1) {
    const prev = ordered[i - 1];
    const next = ordered[i];
    if (next.range.start <= prev.range.end) continue;
    const gap = text.slice(prev.range.end, next.range.start);
    if (gap.includes(' ')) continue;
    if (isProtected({ start: Math.max(0, next.range.start - 1), end: next.range.start + 1 }, protectedSpans)) continue;
    out.push({ index: next.range.start, action: 'INSERT_SPACE', score: 0.96 });
  }
  return out;
}

function mergeSpacingCandidates(
  primary: Array<{ index: number; action: 'INSERT_SPACE' | 'DELETE_SPACE'; score: number }>,
  secondary: Array<{ index: number; action: 'INSERT_SPACE' | 'DELETE_SPACE'; score: number }>
): Array<{ index: number; action: 'INSERT_SPACE' | 'DELETE_SPACE'; score: number }> {
  const merged = new Map<number, { index: number; action: 'INSERT_SPACE' | 'DELETE_SPACE'; score: number }>();
  for (const candidate of [...primary, ...secondary]) {
    const existing = merged.get(candidate.index);
    if (!existing || candidate.score > existing.score) {
      merged.set(candidate.index, candidate);
    }
  }
  return [...merged.values()].sort((a, b) => a.index - b.index);
}

function summarizeGuardrails(decisions: GuardrailDecision[]): GuardrailDecision {
  if (decisions.some((d) => d.decision === 'REJECT')) {
    return { decision: 'REJECT', reasonCodes: ['HAS_REJECT'], score: 0.2 };
  }
  if (decisions.some((d) => d.decision === 'SUGGEST_ONLY')) {
    return { decision: 'SUGGEST_ONLY', reasonCodes: ['HAS_SUGGEST'], score: 0.7 };
  }
  return { decision: 'AUTO_APPLY', reasonCodes: ['ALL_SAFE'], score: 0.95 };
}

function runtimeModelWarnings(modelReady: Record<string, boolean>): string[] {
  return Object.values(modelReady).some(Boolean)
    ? []
    : ['실제 브라우저 ONNX 모델이 없어 heuristic 경로로 동작 중입니다. 배포 자산 또는 models-local 구성을 확인하세요.'];
}

function mergeModelStatus(
  modelReady: Record<string, boolean>,
  modelSource: Record<string, 'local' | 'bundled' | 'remote' | 'missing'>
): {
  ready: Record<string, boolean>;
  source: Record<string, 'local' | 'bundled' | 'remote' | 'missing'>;
} {
  const browserTagger = getBrowserEditTaggerStatus();
  return {
    ready: {
      ...modelReady,
      edit_tagger: browserTagger.ready,
    },
    source: {
      ...modelSource,
      edit_tagger: browserTagger.source,
    },
  };
}

function makeAssetStatus(
  assets: Awaited<ReturnType<typeof loadRuntimeAssets>>,
  kiwiReady: boolean,
  modelReady: Record<string, boolean>,
  modelSource: Record<string, 'local' | 'bundled' | 'remote' | 'missing'>,
  provider: 'wasm' | 'webgpu',
  kiwiError?: string
) {
  const mergedModels = mergeModelStatus(modelReady, modelSource);
  const browserTagger = getBrowserEditTaggerStatus();
  const modelWarnings = runtimeModelWarnings(mergedModels.ready);
  const taggerWarnings = browserTagger.error ? [browserTagger.error] : [];
  return {
    appShellReady: true,
    rulesReady: assets.ready,
    kiwiReady,
    smallModelsReady: Object.values(mergedModels.ready).some(Boolean),
    offlineCapable: assets.ready && kiwiReady,
    runtimeMode: 'browser' as const,
    provider,
    modelReady: mergedModels.ready,
    modelSource: mergedModels.source,
    assetErrors: kiwiError
      ? [...assets.errors, ...modelWarnings, ...taggerWarnings, kiwiError]
      : [...assets.errors, ...modelWarnings, ...taggerWarnings],
  };
}

async function refineProfileWithOnnx(
  text: string,
  base: { profile: Profile; confidence: number }
): Promise<{ profile: Profile; confidence: number; modelUsed: boolean }> {
  const session = await modelManager.getSession('profile_classifier');
  if (!session) return { ...base, modelUsed: false };

  const vec = await runNumericModel(session, makeTextFeatures(text));
  if (!vec || vec.length < PROFILE_LABELS.length) {
    return { ...base, modelUsed: false };
  }

  const logits = vec.slice(0, PROFILE_LABELS.length);
  const pred = argmax(logits);
  const conf = softmaxConfidence(logits);
  const baseIdx = PROFILE_LABELS.indexOf(base.profile);
  const shouldTakeModel = pred.index === baseIdx || conf >= 0.7;

  if (!shouldTakeModel) return { ...base, modelUsed: true };

  return {
    profile: PROFILE_LABELS[pred.index],
    confidence: Math.max(base.confidence, conf),
    modelUsed: true,
  };
}

async function refineSpacingCandidatesWithOnnx(
  text: string,
  profile: Profile,
  candidates: ReturnType<typeof proposeSpacing>
): Promise<{ candidates: ReturnType<typeof proposeSpacing>; modelUsed: boolean }> {
  const session = await modelManager.getSession('spacing_boundary');
  if (!session || candidates.length === 0) return { candidates, modelUsed: false };

  const out = [...candidates];
  let used = false;
  for (let i = 0; i < out.length; i += 1) {
    const c = out[i];
    const vec = await runNumericModel(session, makeBoundaryFeatures(text, c.index, c.score, profileCode(profile)));
    const m = scoreFromVector(vec, c.score);
    const blended = blend(c.score, m, 0.4);
    out[i] = { ...c, score: Math.min(0.99, Math.max(0.01, blended)) };
    used = used || !!vec;
  }
  return { candidates: out, modelUsed: used };
}

async function refineTaggerWithOnnx(
  tags: TaggedToken[],
  profile: Profile
): Promise<{ tags: TaggedToken[]; modelUsed: boolean }> {
  if (tags.length === 0) return { tags, modelUsed: false };

  const browserPredictions = await predictBrowserEditTagger(tags.map((tag) => tag.token));
  if (browserPredictions) {
    const out: TaggedToken[] = [];
    for (let i = 0; i < tags.length; i += 1) {
      const base = tags[i];
      const prediction = browserPredictions[i];
      if (!prediction) {
        out.push(base);
        continue;
      }

      if (prediction.label === 'OPEN_REPLACE') {
        out.push({
          ...base,
          label: 'OPEN_REPLACE',
          confidence: Math.max(base.label === 'OPEN_REPLACE' ? base.confidence : 0, prediction.confidence),
        });
        continue;
      }

      if (base.label === 'OPEN_REPLACE') {
        if (prediction.label === 'KEEP' && prediction.confidence >= 0.8 && base.confidence < 0.95) {
          out.push({
            ...base,
            label: 'KEEP',
            confidence: Math.max(0.7, prediction.confidence),
          });
        } else {
          out.push({
            ...base,
            confidence: Math.max(base.confidence, prediction.confidence * 0.7),
          });
        }
        continue;
      }

      out.push({
        ...base,
        confidence: Math.max(base.confidence, prediction.label === 'KEEP' ? prediction.confidence : base.confidence),
      });
    }
    return { tags: out, modelUsed: true };
  }

  const session = await modelManager.getSession('edit_tagger');
  if (!session) return { tags, modelUsed: false };

  const out: TaggedToken[] = [];
  let used = false;
  for (const t of tags) {
    if (t.label === 'KEEP' && t.confidence >= 0.999) {
      out.push(t);
      continue;
    }

    const vec = await runNumericModel(
      session,
      makeTokenFeatures(t.token).concat(profileCode(profile), t.confidence)
    );
    const openScore = scoreFromVector(vec, t.label === 'OPEN_REPLACE' ? 0.85 : 0.2);
    used = used || !!vec;

    if (t.label === 'OPEN_REPLACE') {
      out.push({ ...t, confidence: Math.max(t.confidence, openScore) });
      continue;
    }

    if (openScore >= 0.86) {
      out.push({ ...t, label: 'OPEN_REPLACE', confidence: openScore });
    } else {
      out.push({ ...t, confidence: Math.max(t.confidence, 1 - openScore) });
    }
  }
  return { tags: out, modelUsed: used };
}

async function refineRerankerWithOnnx(
  text: string,
  profile: Profile,
  candidates: Candidate[]
): Promise<{ candidates: Candidate[]; modelUsed: boolean }> {
  const session = await modelManager.getSession('reranker');
  if (!session || candidates.length === 0) return { candidates, modelUsed: false };

  const out: Candidate[] = [];
  let used = false;
  for (const c of candidates) {
    const vec = await runNumericModel(
      session,
      makeRerankFeatures(
        text,
        c.original,
        c.replacement,
        c.generatorScore,
        c.rerankScore ?? c.finalScore ?? 0.5,
        profileCode(profile)
      )
    );
    const modelScore = scoreFromVector(vec, c.finalScore ?? 0.5);
    const finalScore = blend(c.finalScore ?? 0.5, modelScore, 0.35);
    out.push({ ...c, finalScore, rerankScore: blend(c.rerankScore ?? 0.5, modelScore, 0.25) });
    used = used || !!vec;
  }
  out.sort((a, b) => (b.finalScore ?? 0) - (a.finalScore ?? 0));
  return { candidates: out, modelUsed: used };
}

async function refineGuardrailWithOnnx(
  original: string,
  replacement: string,
  profile: Profile,
  current: GuardrailDecision,
  confidence: number
): Promise<{ decision: GuardrailDecision; modelUsed: boolean }> {
  const session = await modelManager.getSession('guardrail');
  if (!session) return { decision: current, modelUsed: false };

  const ratio = Math.abs(replacement.length - original.length) / Math.max(1, original.length);
  const vec = await runNumericModel(session, [confidence, ratio, profileCode(profile), original.length / 32, replacement.length / 32, 1, 0, 0]);
  if (!vec) return { decision: current, modelUsed: false };

  const safeScore = scoreFromVector(vec, 0.5);
  if (current.decision === 'REJECT') return { decision: current, modelUsed: true };
  if (safeScore < 0.2) {
    return {
      decision: { decision: 'REJECT', reasonCodes: [...current.reasonCodes, 'MODEL_RISK_HIGH'], score: safeScore },
      modelUsed: true,
    };
  }
  if (safeScore < 0.65) {
    return {
      decision: { decision: 'SUGGEST_ONLY', reasonCodes: [...current.reasonCodes, 'MODEL_RISK_MID'], score: safeScore },
      modelUsed: true,
    };
  }
  return {
    decision: {
      decision: current.decision,
      reasonCodes: [...current.reasonCodes, 'MODEL_SAFE'],
      score: Math.max(current.score, safeScore),
    },
    modelUsed: true,
  };
}

async function runPipeline(text: string, cursor: number, mode: 'realtime' | 'inspect'): Promise<PipelineResult> {
  const traces: StageTrace[] = [];
  const allEdits: TextEdit[] = [];
  let allCandidates: Candidate[] = [];
  let lastGuardrail: GuardrailDecision = { decision: 'AUTO_APPLY', reasonCodes: ['INIT'], score: 1 };

  const t1 = now();
  const range = extractRange(text, cursor, mode);
  traces.push(
    makeTrace(
      'Input Range Extractor',
      text,
      {
        clauseText: range.clauseText,
        clauseRange: range.range,
        strategy: mode === 'inspect' ? 'full_text_inspect' : 'cursor_window',
      },
      t1,
      now(),
      '현재 교정해야 할 절만 잘라서 지연시간을 줄입니다.'
    )
  );

  let clause = range.clauseText;
  const assets = await loadRuntimeAssets();

  const t2 = now();
  const protectedOut = detectProtectedSpans(clause);
  traces.push(
    makeTrace('Protected Span Detector', clause, protectedOut, t2, now(), 'URL/숫자/고유명사 같은 구간은 교정에서 제외합니다.')
  );

  const t3 = now();
  const normalized = normalizeText(protectedOut.maskedText);
  const profileHeuristic = classifyProfile(normalized);
  const profileOut = await refineProfileWithOnnx(normalized, profileHeuristic);
  traces.push(
    makeTrace(
      'Profile Classifier',
      normalized,
      {
        profile: profileOut.profile,
        confidence: profileOut.confidence,
        method: profileOut.modelUsed ? 'onnx+heuristic' : 'heuristic',
      },
      t3,
      now(),
      '문장 유형(NORMAL/CHAT/NOISY/MIXED/QUERY)을 정해 이후 정책에 반영합니다.'
    )
  );

  const profile: Profile = profileOut.profile;

  const t4 = now();
  const ruleEdits = runRuleCorrector(clause, protectedOut.protectedSpans, assets.surfaceFixRules);
  traces.push(makeTrace('Rule Corrector', clause, ruleEdits, t4, now(), '확실한 교정 규칙을 먼저 적용합니다.'));
  if (ruleEdits.length) {
    clause = applyEdits(clause, ruleEdits);
    allEdits.push(...ruleEdits);
  }

  const t41 = now();
  const phraseEdits = runPhrasePreNormalizer(clause, protectedOut.protectedSpans, assets.phraseRules);
  traces.push(
    makeTrace(
      'Phrase Pre-normalizer',
      clause,
      phraseEdits,
      t41,
      now(),
      '고정밀 구문 교정 메모리를 먼저 적용해 후보 공간을 줄입니다.'
    )
  );
  if (phraseEdits.length) {
    clause = applyEdits(clause, phraseEdits);
    allEdits.push(...phraseEdits);
  }

  const t5 = now();
  const spacingCandidatesRaw = proposeSpacing(clause, profile, protectedOut.protectedSpans);
  const preSpacingKiwiAnalysis = await analyzeCanonicalTokens(clause);
  const kiwiSpacingCandidates = kiwiSpacingFallback(clause, preSpacingKiwiAnalysis, protectedOut.protectedSpans);
  const spacingSeeded = mergeSpacingCandidates(spacingCandidatesRaw, kiwiSpacingCandidates);
  const spacingModelOut = await refineSpacingCandidatesWithOnnx(clause, profile, spacingSeeded);
  const spacingCandidates = spacingModelOut.candidates;
  traces.push(
    makeTrace(
      'Spacing Candidate Generator',
      clause,
      {
        spacedText: clause,
        candidates: spacingCandidates,
        method: spacingModelOut.modelUsed ? 'onnx+heuristic' : 'heuristic',
      },
      t5,
      now(),
      '띄어쓰기 후보 경계를 생성합니다(Kiwi 대체 인터페이스).'
    )
  );

  const t6 = now();
  const spacingAccepted = classifySpacingBoundaries(clause, spacingCandidates, profile);
  const t65 = now();
  let kiwiAnalysis = preSpacingKiwiAnalysis;

  const spacingAcceptedFinal =
    spacingAccepted.length > 0 ? spacingAccepted : kiwiSpacingCandidates;
  const spacingEdits = spacingToEdits(spacingAcceptedFinal);
  traces.push(
    makeTrace('Spacing Boundary Classifier', clause, spacingAcceptedFinal, t6, now(), '후보 중 적용할 띄어쓰기만 선별합니다.')
  );
  traces.push(
    makeTrace('Spacing Edit Converter', clause, spacingEdits, t6, now(), '채택된 띄어쓰기 후보를 실제 수정안으로 바꿉니다.')
  );
  if (spacingEdits.length) {
    clause = applyEdits(clause, spacingEdits);
    allEdits.push(...spacingEdits);
    kiwiAnalysis = await analyzeCanonicalTokens(clause);
  }
  traces.push(
    makeTrace(
      'Kiwi Canonicalizer',
      clause,
      {
        ready: kiwiAnalysis.ready,
        version: kiwiAnalysis.version,
        tokenCount: kiwiAnalysis.tokensByRange.size,
        error: kiwiAnalysis.error,
      },
      t65,
      now(),
      'Kiwi wasm으로 lemma/POS/활용 슬롯을 추출해 후보 생성의 기준 단위를 canonical state로 올립니다.'
    )
  );

  const t7 = now();
  const taggedBase = runEditTagger(clause, protectedOut.protectedSpans, assets, kiwiAnalysis);
  const taggedOut = await refineTaggerWithOnnx(taggedBase, profile);
  const tagged = taggedOut.tags;
  traces.push(
    makeTrace(
      'Edit Tagger (Browser)',
      clause,
      tagged,
      t7,
      now(),
      '토큰별 KEEP/OPEN_REPLACE 등 편집 태그를 예측합니다.'
    )
  );

  const t8 = now();
  const generated = await generateCandidates(clause, tagged, assets, kiwiAnalysis);
  traces.push(
    makeTrace('Open Candidate Generation', clause, groupCandidatesForTrace(generated), t8, now(), 'OPEN_REPLACE 토큰에 대한 교체 후보를 만듭니다.')
  );

  const t82 = now();
  const verified = verifyCandidates(clause, generated, assets, kiwiAnalysis);
  traces.push(
    makeTrace(
      'CandidateVerifier',
      clause,
      groupVerifierForTrace(generated, verified),
      t82,
      now(),
      'POS/활용 슬롯/오타 거리 기준으로 구조적으로 말이 되는 후보만 남깁니다.'
    )
  );

  const tagConf: Record<string, number> = {};
  for (const t of tagged) {
    tagConf[`${t.range.start}:${t.range.end}`] = t.confidence;
  }

  const t9 = now();
  const rerankedBase = rerankCandidates(clause, profile, verified, tagConf);
  const rerankedOut = await refineRerankerWithOnnx(clause, profile, rerankedBase);
  const reranked = rerankedOut.candidates;
  traces.push(
    makeTrace(
      'Reranker',
      clause,
      groupRerankerForTrace(reranked),
      t9,
      now(),
      '후보 문맥 점수를 계산해 우선순위를 정합니다.'
    )
  );
  allCandidates = reranked;

  const t10 = now();
  const best = groupBestBySpan(reranked);
  const openEdits: Array<{ edit: TextEdit; guardrail: GuardrailDecision; modelUsed: boolean }> = [];
  for (const c of best) {
    const edit: TextEdit = {
      stage: 'RERANKER',
      range: c.span,
      sourceText: c.original,
      replacement: c.replacement,
      editType: 'OPEN_REPLACE',
      confidence: c.finalScore ?? 0.5,
      autoApplicable: false,
      reasonTag: c.source,
    };
    const guardrailBase = guardrailDecision(
      c.original,
      c.replacement,
      c.span,
      profile,
      protectedOut.protectedSpans,
      c.finalScore ?? 0.5
    );
    const guardrailOut = await refineGuardrailWithOnnx(
      c.original,
      c.replacement,
      profile,
      guardrailBase,
      c.finalScore ?? 0.5
    );
    openEdits.push({ edit, guardrail: guardrailOut.decision, modelUsed: guardrailOut.modelUsed });
  }
  traces.push(
    makeTrace(
      'Guardrail',
      clause,
      openEdits.map((o) => ({
        original: o.edit.sourceText,
        replacement: o.edit.replacement,
        guardrail: o.guardrail,
      })),
      t10,
      now(),
      '과교정/의미변경 위험을 검사해 자동 적용 여부를 제한합니다.'
    )
  );

  const t11 = now();
  const decisions = openEdits.map((o) => {
    const decision = policyForEdit(o.edit, o.guardrail);
    return { edit: { ...o.edit, autoApplicable: decision === 'AUTO_APPLY' }, decision, guardrail: o.guardrail };
  });

  const policy = runPolicyEngine(decisions.map((d) => ({ edit: d.edit, decision: d.decision })));
  traces.push(
    makeTrace(
      'Policy Engine',
      clause,
      decisions.map((d) => ({
        original: d.edit.sourceText,
        replacement: d.edit.replacement,
        decision: d.decision,
      })),
      t11,
      now(),
      '자동 적용/제안/차단을 최종 분류합니다.'
    )
  );

  lastGuardrail = summarizeGuardrails(decisions.map((d) => d.guardrail));

  const t12 = now();
  const clauseAfterPolicy = applyEdits(clause, policy.autoApply.filter((e) => e.editType === 'OPEN_REPLACE'));
  const suggestedClause = applyEdits(
    clauseAfterPolicy,
    policy.suggestOnly.filter((e) => e.editType === 'OPEN_REPLACE')
  );
  const corrected = `${text.slice(0, range.range.start)}${clauseAfterPolicy}${text.slice(range.range.end)}`;
  const suggestedText = `${text.slice(0, range.range.start)}${suggestedClause}${text.slice(range.range.end)}`;
  const finalEdits = [...allEdits, ...policy.suggestOnly, ...policy.reject, ...policy.autoApply.filter((e) => e.editType === 'OPEN_REPLACE')];
  traces.push(
    makeTrace(
      'Final Output',
      text,
      {
        finalClause: clauseAfterPolicy,
        autoCount: policy.autoApply.length,
        suggestCount: policy.suggestOnly.length,
        rejectCount: policy.reject.length,
      },
      t12,
      now(),
      '정책에 따라 자동 적용 수정만 반영해 최종 교정문을 만듭니다.'
    )
  );

  const modelReady = modelManager.getModelReadyMap();
  const modelSource = modelManager.getModelSourceMap();
  const kiwiStatus = getKiwiRuntimeStatus();

  return {
    original: text,
    corrected,
    suggestedText,
    profile,
    protectedSpans: protectedOut.protectedSpans,
    edits: finalEdits,
    candidates: allCandidates,
    guardrail: lastGuardrail,
    traces,
    assetStatus: makeAssetStatus(
      assets,
      kiwiAnalysis.ready,
      modelReady,
      modelSource,
      modelManager.getExecutionProvider(),
      kiwiStatus.error
    ),
  };
}

self.onmessage = async (evt: MessageEvent<WorkerRequest>): Promise<void> => {
  const msg = evt.data;

  try {
    if (msg.type === 'INIT') {
      await modelManager.init(msg.provider);
      const assets = await loadRuntimeAssets();
      void ensureKiwiRuntime();
      void warmupBrowserEditTagger();
      const kiwiStatus = getKiwiRuntimeStatus();
      const modelReady = modelManager.getModelReadyMap();
      const modelSource = modelManager.getModelSourceMap();
      const response: WorkerResponse = {
        type: 'INIT_DONE',
        provider: modelManager.getExecutionProvider(),
        modelReady,
        modelSource,
        assetStatus: makeAssetStatus(
          assets,
          kiwiStatus.ready,
          modelReady,
          modelSource,
          modelManager.getExecutionProvider(),
          kiwiStatus.error
        ),
      };
      self.postMessage(response);
      return;
    }

    if (msg.type === 'WARMUP') {
      await modelManager.warmup(msg.models.filter((name) => name !== 'edit_tagger'));
      const assets = await loadRuntimeAssets();
      const kiwi = await ensureKiwiRuntime();
      if (msg.models.includes('edit_tagger')) {
        await warmupBrowserEditTagger();
      }
      const kiwiStatus = getKiwiRuntimeStatus();
      const modelReady = modelManager.getModelReadyMap();
      const modelSource = modelManager.getModelSourceMap();
      const response: WorkerResponse = {
        type: 'WARMUP_DONE',
        provider: modelManager.getExecutionProvider(),
        modelReady,
        modelSource,
        assetStatus: makeAssetStatus(
          assets,
          kiwiStatus.ready || !!kiwi,
          modelReady,
          modelSource,
          modelManager.getExecutionProvider(),
          kiwiStatus.error
        ),
      };
      self.postMessage(response);
      return;
    }

    if (msg.type === 'RUN_PIPELINE') {
      const result = await runPipeline(msg.text, msg.cursor, msg.mode);
      const response: WorkerResponse = { type: 'PIPELINE_RESULT', result };
      self.postMessage(response);
      return;
    }

    if (msg.type === 'RUN_STAGE') {
      const response: WorkerResponse = {
        type: 'STAGE_RESULT',
        stage: msg.stage,
        trace: {
          stageName: msg.stage,
          inputText: '',
          output: msg.payload,
          latencyMs: 0,
          why: '개별 stage 디버그 실행',
        },
      };
      self.postMessage(response);
    }
  } catch (error) {
    const response: WorkerResponse = {
      type: 'ERROR',
      message: error instanceof Error ? error.message : 'Unknown worker error',
    };
    self.postMessage(response);
  }
};
