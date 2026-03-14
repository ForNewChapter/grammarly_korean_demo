export type InputMode = 'realtime' | 'inspect';
export type Profile = 'NORMAL' | 'CHAT' | 'NOISY' | 'MIXED' | 'QUERY';

export interface TextRange {
  start: number;
  end: number;
}

export interface ProtectedSpan {
  range: TextRange;
  kind:
    | 'URL'
    | 'EMAIL'
    | 'PHONE'
    | 'NUMBER'
    | 'DATETIME'
    | 'CODE'
    | 'ENTITY'
    | 'USER_DICT'
    | 'HASHTAG'
    | 'MENTION';
  text: string;
  placeholder: string;
}

export interface TextEdit {
  stage: 'RULE' | 'PHRASE' | 'SPACING' | 'EDIT_TAGGER' | 'CANDIDATE_GEN' | 'RERANKER' | 'GUARDRAIL';
  range: TextRange;
  sourceText: string;
  replacement: string;
  editType:
    | 'SPELL'
    | 'SPACE_INSERT'
    | 'SPACE_DELETE'
    | 'JOSA'
    | 'EOMI'
    | 'PUNCT'
    | 'OPEN_REPLACE'
    | 'DELETE'
    | 'MERGE'
    | 'SPLIT';
  confidence: number;
  autoApplicable: boolean;
  reasonTag: string;
}

export interface Candidate {
  span: TextRange;
  original: string;
  replacement: string;
  source: string;
  generatorScore: number;
  verifyScore?: number;
  rerankScore?: number;
  finalScore?: number;
  familyLemma?: string;
  posHint?: string;
  slotHints?: string[];
  contextHints?: string[];
  typoDistance?: number;
}

export interface StageTrace {
  stageName: string;
  inputText: string;
  output: unknown;
  latencyMs: number;
  why: string;
}

export interface GuardrailDecision {
  decision: 'AUTO_APPLY' | 'SUGGEST_ONLY' | 'REJECT';
  reasonCodes: string[];
  score: number;
}

export interface PipelineResult {
  original: string;
  corrected: string;
  suggestedText?: string;
  profile: Profile;
  protectedSpans: ProtectedSpan[];
  edits: TextEdit[];
  candidates: Candidate[];
  guardrail: GuardrailDecision;
  traces: StageTrace[];
  assetStatus: {
    appShellReady: boolean;
    rulesReady: boolean;
    kiwiReady?: boolean;
    smallModelsReady: boolean;
    offlineCapable: boolean;
    provider: 'wasm' | 'webgpu';
    modelReady: Record<string, boolean>;
    assetErrors?: string[];
  };
}

export interface PipelineContext {
  mode: InputMode;
  fullText: string;
  cursor: number;
}

export interface StageRunnerOutput<T> {
  value: T;
  trace: StageTrace;
}
