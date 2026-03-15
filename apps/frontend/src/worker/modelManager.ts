import * as ort from 'onnxruntime-web';
import { resolveAssetUrl } from '../cache/assetUrl';

const MODEL_PATHS = {
  profile_classifier: 'profile_classifier.onnx',
  spacing_boundary: 'spacing_boundary.onnx',
  edit_tagger: 'edit_tagger.onnx',
  reranker: 'reranker.onnx',
  guardrail: 'guardrail.onnx',
} as const;

type ModelName = keyof typeof MODEL_PATHS;
type ModelSource = 'local' | 'bundled' | 'remote' | 'missing';

function modelUrlCandidates(fileName: string): Array<{ url: string; source: Exclude<ModelSource, 'missing'> }> {
  return [
    { url: resolveAssetUrl(`/assets/models-local/${fileName}`), source: 'local' },
    { url: resolveAssetUrl(`/assets/models/${fileName}`), source: 'bundled' },
  ];
}

export class ModelManager {
  private provider: 'wasm' | 'webgpu' = 'wasm';
  private sessions = new Map<ModelName, ort.InferenceSession | null>();
  private modelReady: Record<string, boolean> = {};
  private modelSource: Record<string, ModelSource> = {};

  async init(provider: 'wasm' | 'webgpu'): Promise<void> {
    this.provider = provider;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    for (const name of Object.keys(MODEL_PATHS) as ModelName[]) {
      this.sessions.set(name, null);
      this.modelReady[name] = false;
      this.modelSource[name] = 'missing';
    }
  }

  async warmup(modelNames: string[]): Promise<void> {
    for (const m of modelNames) {
      if (m in MODEL_PATHS) {
        await this.getSession(m as ModelName);
      }
    }
  }

  getExecutionProvider(): 'wasm' | 'webgpu' {
    return this.provider;
  }

  getModelReadyMap(): Record<string, boolean> {
    return { ...this.modelReady };
  }

  getModelSourceMap(): Record<string, ModelSource> {
    return { ...this.modelSource };
  }

  async getSession(modelName: ModelName): Promise<ort.InferenceSession | null> {
    const existing = this.sessions.get(modelName);
    if (existing !== undefined && existing !== null) return existing;

    const candidates = modelUrlCandidates(MODEL_PATHS[modelName]);
    for (const candidate of candidates) {
      try {
        const head = await fetch(candidate.url, { method: 'HEAD' });
        if (!head.ok) {
          continue;
        }
        const contentLength = Number(head.headers.get('content-length') ?? '0');
        if (!Number.isFinite(contentLength) || contentLength < 1024) {
          continue;
        }
        const session = await ort.InferenceSession.create(candidate.url, {
          executionProviders: [this.provider],
        });
        this.sessions.set(modelName, session);
        this.modelReady[modelName] = true;
        this.modelSource[modelName] = candidate.source;
        return session;
      } catch {
        continue;
      }
    }

    this.modelReady[modelName] = false;
    this.modelSource[modelName] = 'missing';
    return null;
  }
}
