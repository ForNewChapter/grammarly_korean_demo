import type { InputMode, PipelineResult, StageTrace } from './types';
import type { WorkerRequest, WorkerResponse } from '../worker/protocol';

interface PendingRequest<T> {
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

type RuntimeModelSource = 'local' | 'bundled' | 'remote' | 'backend' | 'missing';

function normalizeApiBaseUrl(value?: string): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  return trimmed.endsWith('/') ? trimmed.slice(0, -1) : trimmed;
}

export interface InitResult {
  provider: 'wasm' | 'webgpu';
  modelReady: Record<string, boolean>;
  assetStatus: {
    appShellReady: boolean;
    rulesReady: boolean;
    kiwiReady?: boolean;
    smallModelsReady: boolean;
    offlineCapable: boolean;
    runtimeMode?: 'browser' | 'backend';
    modelSource?: Record<string, RuntimeModelSource>;
    assetErrors?: string[];
    backend?: {
      apiBaseUrl: string;
      backendName?: string | null;
      backendBranch?: string | null;
      backendVersion?: string | null;
      engineReady?: boolean;
      engineInitializing?: boolean;
      engineError?: string | null;
    };
  };
}

export interface RuntimeStatus {
  provider: 'wasm' | 'webgpu';
  modelReady: Record<string, boolean>;
  assetStatus: InitResult['assetStatus'];
}

interface BackendHealthResponse {
  ok?: boolean;
  backendName?: string | null;
  backendBranch?: string | null;
  backendVersion?: string | null;
  engineReady?: boolean;
  engineInitializing?: boolean;
  engineError?: string | null;
}

export class PipelineOrchestrator {
  private worker: Worker | null = null;
  private pendingInit: PendingRequest<InitResult> | null = null;
  private pendingPipeline: PendingRequest<PipelineResult> | null = null;
  private pendingWarmup: PendingRequest<RuntimeStatus | null> | null = null;
  private stageListener: ((trace: StageTrace) => void) | null = null;
  private useLocalApi = false;
  private provider: 'wasm' | 'webgpu' = 'wasm';
  private backendHealth: BackendHealthResponse | null = null;
  private apiBaseUrl =
    typeof window !== 'undefined'
      ? normalizeApiBaseUrl(
          new URLSearchParams(window.location.search).get('apiBase') ?? import.meta.env.VITE_API_BASE_URL
        )
      : normalizeApiBaseUrl(import.meta.env.VITE_API_BASE_URL);
  private allowBrowserFallback =
    typeof window !== 'undefined'
      ? new URLSearchParams(window.location.search).get('browser') === '1' ||
        import.meta.env.VITE_ALLOW_BROWSER_FALLBACK === '1'
      : import.meta.env.VITE_ALLOW_BROWSER_FALLBACK === '1';
  private preferLocalApi =
    typeof window !== 'undefined' &&
    (new URLSearchParams(window.location.search).get('localApi') === '1' ||
      import.meta.env.VITE_PREFER_LOCAL_API === '1');

  constructor() {
    if (!this.apiBaseUrl && this.allowBrowserFallback) {
      this.ensureWorker();
    }
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    this.worker = new Worker(new URL('../worker/inference.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (evt: MessageEvent<WorkerResponse>) => {
      const msg = evt.data;
      if (msg.type === 'INIT_DONE' && this.pendingInit) {
        this.pendingInit.resolve({ provider: msg.provider, modelReady: msg.modelReady, assetStatus: msg.assetStatus });
        this.pendingInit = null;
      }
      if (msg.type === 'PIPELINE_RESULT' && this.pendingPipeline) {
        this.pendingPipeline.resolve(msg.result);
        this.pendingPipeline = null;
      }
      if (msg.type === 'WARMUP_DONE' && this.pendingWarmup) {
        this.pendingWarmup.resolve({
          provider: msg.provider,
          modelReady: msg.modelReady,
          assetStatus: msg.assetStatus,
        });
        this.pendingWarmup = null;
      }
      if (msg.type === 'STAGE_RESULT' && this.stageListener) {
        this.stageListener(msg.trace);
      }
      if (msg.type === 'ERROR') {
        this.pendingInit?.reject(new Error(msg.message));
        this.pendingWarmup?.reject(new Error(msg.message));
        this.pendingPipeline?.reject(new Error(msg.message));
        this.pendingInit = null;
        this.pendingWarmup = null;
        this.pendingPipeline = null;
      }
    };
    return this.worker;
  }

  private resolveApiUrl(path: string): string {
    if (!this.apiBaseUrl) return path;
    return `${this.apiBaseUrl}${path}`;
  }

  setStageListener(listener: ((trace: StageTrace) => void) | null): void {
    this.stageListener = listener;
  }

  private async checkLocalApi(): Promise<BackendHealthResponse | null> {
    try {
      const res = await fetch(this.resolveApiUrl('/api/health'), { method: 'GET', cache: 'no-store' });
      if (!res.ok) return null;
      const body = (await res.json()) as BackendHealthResponse;
      return body?.ok ? body : null;
    } catch {
      return null;
    }
  }

  private mapApiTrace(raw: any): StageTrace {
    const stageName = typeof raw?.stageName === 'string' ? raw.stageName : 'UnknownStage';
    const cleanName = stageName.replace(/^\d+(?:-\d+)?\.\s*/, '');
    const whyMap: Array<[RegExp, string]> = [
      [/Input Range/i, '현재 입력에서 검사 범위를 추출합니다.'],
      [/Protected Span/i, '보호 구간(URL/숫자/고유명사)을 잠급니다.'],
      [/Profile/i, '문장 프로파일을 분류합니다.'],
      [/Rule/i, '확정 규칙 교정을 수행합니다.'],
      [/Spacing Candidate/i, '띄어쓰기 후보를 생성합니다.'],
      [/Boundary/i, '띄어쓰기 후보를 선별합니다.'],
      [/Edit Tagger/i, '오류 토큰을 탐지/태깅합니다.'],
      [/Candidate Generation/i, '열린 치환 후보를 생성합니다.'],
      [/Reranker/i, '후보를 문맥 점수로 재정렬합니다.'],
      [/Guardrail/i, '과교정/의미변경 위험을 검증합니다.'],
      [/Policy/i, '자동/제안/차단 정책을 결정합니다.'],
      [/Final Output/i, '최종 문장을 합성합니다.'],
    ];
    const why = whyMap.find(([re]) => re.test(stageName))?.[1] ?? '단계 실행 결과입니다.';
    return {
      stageName: cleanName,
      inputText: raw?.inputText ?? '',
      output: raw?.outputArtifacts ?? raw?.output ?? null,
      latencyMs: Number(raw?.latencyMs ?? 0),
      why,
    };
  }

  private mapApiResult(api: any, originalText: string): PipelineResult {
    const traces = Array.isArray(api?.traces) ? api.traces.map((t: any) => this.mapApiTrace(t)) : [];
    const profileStage = traces.find((t: StageTrace) => /Profile Classifier/i.test(t.stageName));
    const profile = (profileStage?.output as any)?.profile ?? 'NORMAL';

    const protectedStage = traces.find((t: StageTrace) => /Protected Span Detector/i.test(t.stageName));
    const protectedSpans = ((protectedStage?.output as any)?.protected ?? []) as PipelineResult['protectedSpans'];

    const rerankStage = traces.find((t: StageTrace) => /Reranker/i.test(t.stageName));
    const rerankGroups = Array.isArray(rerankStage?.output) ? (rerankStage?.output as any[]) : [];
    const candidates = rerankGroups.flatMap((g) => {
      const ranked = Array.isArray(g?.ranked) ? g.ranked : [];
      return ranked.map((r: any) => ({
        span: g.span,
        original: g.original,
        replacement: r.replacement,
        source: r.source ?? 'ORIGINAL',
        generatorScore: Number(r.generatorScore ?? 0),
        rerankScore: Number(r.rerankScore ?? 0),
        finalScore: Number(r.finalScore ?? 0),
      }));
    });

    const decisions = Array.isArray(api?.decisions) ? api.decisions : [];
    const edits = decisions.map((d: any) => ({
      stage: d.stage,
      range: d.range,
      sourceText: d.sourceText,
      replacement: d.replacement,
      editType: d.editType,
      confidence: Number(d.confidence ?? 0),
      autoApplicable: d.decision === 'AUTO_APPLY' || !!d.autoApplicable,
      reasonTag: d.reasonTag ?? '',
    }));

    const hasReject = decisions.some((d: any) => d.decision === 'REJECT');
    const hasSuggest = decisions.some((d: any) => d.decision === 'SUGGEST_ONLY');
    const guardrail = hasReject
      ? { decision: 'REJECT' as const, reasonCodes: ['HAS_REJECT'], score: 0.2 }
      : hasSuggest
        ? { decision: 'SUGGEST_ONLY' as const, reasonCodes: ['HAS_SUGGEST'], score: 0.7 }
        : { decision: 'AUTO_APPLY' as const, reasonCodes: ['ALL_SAFE'], score: 0.95 };

    return {
      original: originalText,
      corrected: api?.finalText ?? originalText,
      suggestedText: api?.suggestedText ?? api?.finalText ?? originalText,
      profile,
      protectedSpans,
      edits,
      candidates,
      guardrail,
      traces,
      assetStatus: {
        appShellReady: true,
        rulesReady: true,
        smallModelsReady: true,
        offlineCapable: false,
        runtimeMode: 'backend',
        provider: this.provider,
        modelReady: {
          kiwi: true,
          kobert: true,
          koelectra: true,
          kobert_mlm: true,
        },
        backend: this.apiBaseUrl
          ? {
              apiBaseUrl: this.apiBaseUrl,
              backendName: this.backendHealth?.backendName ?? null,
              backendBranch: this.backendHealth?.backendBranch ?? null,
              backendVersion: this.backendHealth?.backendVersion ?? null,
              engineReady: true,
              engineInitializing: false,
              engineError: null,
            }
          : undefined,
      },
    };
  }

  async init(provider: 'wasm' | 'webgpu'): Promise<InitResult> {
    this.provider = provider;
    if (!this.apiBaseUrl && !this.preferLocalApi && !this.allowBrowserFallback) {
      throw new Error('백엔드 연결이 필요합니다. 공유 링크의 ?apiBase=... 값을 확인하세요.');
    }
    const mustUseApi = !!this.apiBaseUrl || this.preferLocalApi;
    const apiHealth = mustUseApi ? await this.checkLocalApi() : null;
    const hasApi = !!apiHealth;
    if (mustUseApi && !apiHealth) {
      throw new Error(
        this.apiBaseUrl
          ? `Cloud Run API unavailable: ${this.resolveApiUrl('/api/health')}`
          : 'Local API unavailable: /api/health'
      );
    }
    this.useLocalApi = hasApi;
    this.backendHealth = apiHealth;
    if (hasApi) {
      return {
        provider,
        modelReady: {
          kiwi: true,
          kobert: true,
          koelectra: true,
          kobert_mlm: true,
        },
        assetStatus: {
          appShellReady: true,
          rulesReady: true,
          smallModelsReady: true,
          offlineCapable: false,
          runtimeMode: 'backend',
          modelSource: {
            kiwi: 'backend',
            kobert: 'backend',
            koelectra: 'backend',
            kobert_mlm: 'backend',
          },
          backend: {
            apiBaseUrl: this.resolveApiUrl(''),
            backendName: apiHealth?.backendName ?? null,
            backendBranch: apiHealth?.backendBranch ?? null,
            backendVersion: apiHealth?.backendVersion ?? null,
            engineReady: apiHealth?.engineReady,
            engineInitializing: apiHealth?.engineInitializing,
            engineError: apiHealth?.engineError ?? null,
          },
        },
      };
    }

    return new Promise((resolve, reject) => {
      this.pendingInit = { resolve, reject };
      const req: WorkerRequest = { type: 'INIT', provider };
      this.ensureWorker().postMessage(req);
    });
  }

  warmup(models: string[]): Promise<RuntimeStatus | null> {
    if (this.useLocalApi) return Promise.resolve(null);
    return new Promise((resolve, reject) => {
      this.pendingWarmup = { resolve, reject };
      const req: WorkerRequest = { type: 'WARMUP', models };
      this.ensureWorker().postMessage(req);
    });
  }

  runPipeline(text: string, cursor: number, mode: InputMode): Promise<PipelineResult> {
    if (this.useLocalApi) {
      return fetch(this.resolveApiUrl('/api/correct'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, cursor, mode }),
      })
        .then((res) => {
          if (!res.ok) throw new Error(`Local API request failed: ${res.status}`);
          return res.json();
        })
        .then((api) => this.mapApiResult(api, text));
    }

    return new Promise((resolve, reject) => {
      this.pendingPipeline = { resolve, reject };
      const req: WorkerRequest = { type: 'RUN_PIPELINE', text, cursor, mode };
      this.ensureWorker().postMessage(req);
    });
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
  }
}
