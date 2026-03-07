import * as ort from 'onnxruntime-web';

const MODEL_URLS = {
  profile_classifier: '/assets/models/profile_classifier.onnx',
  spacing_boundary: '/assets/models/spacing_boundary.onnx',
  edit_tagger: '/assets/models/edit_tagger.onnx',
  reranker: '/assets/models/reranker.onnx',
  guardrail: '/assets/models/guardrail.onnx',
} as const;

type ModelName = keyof typeof MODEL_URLS;

export class ModelManager {
  private provider: 'wasm' | 'webgpu' = 'wasm';
  private sessions = new Map<ModelName, ort.InferenceSession | null>();
  private modelReady: Record<string, boolean> = {};

  async init(provider: 'wasm' | 'webgpu'): Promise<void> {
    this.provider = provider;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    for (const name of Object.keys(MODEL_URLS) as ModelName[]) {
      this.sessions.set(name, null);
      this.modelReady[name] = false;
    }
  }

  async warmup(modelNames: string[]): Promise<void> {
    for (const m of modelNames) {
      if (m in MODEL_URLS) {
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

  async getSession(modelName: ModelName): Promise<ort.InferenceSession | null> {
    const existing = this.sessions.get(modelName);
    if (existing !== undefined && existing !== null) return existing;

    const url = MODEL_URLS[modelName];
    try {
      const head = await fetch(url, { method: 'HEAD' });
      if (!head.ok) {
        this.modelReady[modelName] = false;
        return null;
      }
      const session = await ort.InferenceSession.create(url, {
        executionProviders: [this.provider],
      });
      this.sessions.set(modelName, session);
      this.modelReady[modelName] = true;
      return session;
    } catch {
      this.modelReady[modelName] = false;
      return null;
    }
  }
}
