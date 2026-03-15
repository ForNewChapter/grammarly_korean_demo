import {
  AutoModelForTokenClassification,
  AutoTokenizer,
  env as hfEnv,
  Tensor,
} from '@huggingface/transformers';
import { resolveAssetUrl } from '../cache/assetUrl';

type BrowserModelSource = 'local' | 'bundled' | 'remote' | 'missing';
type BrowserTaggerLabel = 'KEEP' | 'SPACE_FIX' | 'OPEN_REPLACE';

interface BrowserEditTaggerRoot {
  root: string;
  source: Exclude<BrowserModelSource, 'missing'>;
}

interface RemoteRootConfig {
  modelId: string;
  remoteHost: string;
  remotePathTemplate: string;
}

interface BrowserEditTaggerRuntime {
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
  model: Awaited<ReturnType<typeof AutoModelForTokenClassification.from_pretrained>>;
  labels: BrowserTaggerLabel[];
  clsTokenId: number;
  sepTokenId: number;
  maxLength: number;
  source: Exclude<BrowserModelSource, 'missing'>;
  root: string;
  dtype: 'fp32' | 'q8';
}

export interface BrowserEditTaggerStatus {
  ready: boolean;
  source: BrowserModelSource;
  root?: string;
  dtype?: 'fp32' | 'q8';
  error?: string;
}

export interface BrowserEditTaggerPrediction {
  label: BrowserTaggerLabel;
  confidence: number;
  logits: number[];
}

const MANIFEST_URL = resolveAssetUrl('/assets/browser_model_manifest.json');
const DEFAULT_ROOTS: BrowserEditTaggerRoot[] = [
  { root: resolveAssetUrl('/assets/browser-models-local/edit_tagger_v2'), source: 'local' },
  { root: resolveAssetUrl('/assets/browser-models/edit_tagger_v2'), source: 'bundled' },
];

let runtimePromise: Promise<BrowserEditTaggerRuntime | null> | null = null;
let runtimeState: BrowserEditTaggerRuntime | null = null;
let runtimeStatus: BrowserEditTaggerStatus = {
  ready: false,
  source: 'missing',
};

function classifySource(root: string): Exclude<BrowserModelSource, 'missing'> {
  if (/\/browser-models-local\//.test(root)) return 'local';
  if (/^https?:\/\//.test(root)) {
    const currentOrigin = typeof globalThis.location?.origin === 'string' ? globalThis.location.origin : '';
    return root.startsWith(currentOrigin) ? 'bundled' : 'remote';
  }
  return 'bundled';
}

function configureTransformersEnv(): void {
  hfEnv.allowLocalModels = true;
  hfEnv.allowRemoteModels = true;
  const wasmBackend = hfEnv.backends?.onnx?.wasm;
  if (wasmBackend) {
    wasmBackend.wasmPaths = resolveAssetUrl('/assets/transformers/');
  }
}

function parseRemoteRoot(root: string): RemoteRootConfig {
  const url = new URL(root);
  const modelId = url.pathname.replace(/^\/+/, '').replace(/\/+$/, '');
  if (!modelId) {
    throw new Error(`Remote browser model root is missing a model id: ${root}`);
  }
  return {
    modelId,
    remoteHost: `${url.origin}/`,
    remotePathTemplate: '{model}/',
  };
}

async function loadManifestRoots(): Promise<BrowserEditTaggerRoot[]> {
  try {
    const response = await fetch(MANIFEST_URL);
    if (!response.ok) return DEFAULT_ROOTS;
    const payload = (await response.json()) as { editTaggerRoots?: string[] };
    const roots = (payload.editTaggerRoots ?? [])
      .map((root) => root?.trim())
      .filter(Boolean)
      .map((root) => ({
        root: /^https?:\/\//.test(root) ? root : resolveAssetUrl(root),
        source: classifySource(root),
      }));
    return roots.length ? roots : DEFAULT_ROOTS;
  } catch {
    return DEFAULT_ROOTS;
  }
}

function softmaxConfidence(logits: number[]): number {
  if (!logits.length) return 0;
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  if (sum <= 0) return 0;
  return Math.max(...exps) / sum;
}

function toBigIntTensor(values: number[]): Tensor {
  return new Tensor('int64', BigInt64Array.from(values.map((value) => BigInt(value))), [1, values.length]);
}

async function createRuntime(): Promise<BrowserEditTaggerRuntime | null> {
  configureTransformersEnv();

  const roots = await loadManifestRoots();
  let lastError = 'No browser edit tagger model found';

  for (const candidate of roots) {
    const defaultAllowLocalModels = hfEnv.allowLocalModels;
    const defaultAllowRemoteModels = hfEnv.allowRemoteModels;
    const defaultRemoteHost = hfEnv.remoteHost;
    const defaultRemotePathTemplate = hfEnv.remotePathTemplate;
    try {
      let modelRef = candidate.root;
      if (candidate.source === 'remote') {
        const remoteConfig = parseRemoteRoot(candidate.root);
        modelRef = remoteConfig.modelId;
        hfEnv.allowLocalModels = false;
        hfEnv.allowRemoteModels = true;
        hfEnv.remoteHost = remoteConfig.remoteHost;
        hfEnv.remotePathTemplate = remoteConfig.remotePathTemplate;
      }

      const configResponse = await fetch(`${candidate.root}/config.json`);
      if (!configResponse.ok) {
        lastError = `${candidate.root}/config.json -> ${configResponse.status}`;
        continue;
      }

      const tokenizer = await AutoTokenizer.from_pretrained(modelRef, {
        local_files_only: candidate.source !== 'remote',
      });

      for (const dtype of ['fp32', 'q8'] as const) {
        try {
          const model = await AutoModelForTokenClassification.from_pretrained(modelRef, {
            local_files_only: candidate.source !== 'remote',
            dtype,
          });
          const specials = (await tokenizer('', { return_tensor: false })) as { input_ids: number[] };
          const config = model.config as {
            id2label?: Record<string, string>;
            max_position_embeddings?: number;
          };
          const labels = Object.values(config.id2label ?? {}) as BrowserTaggerLabel[];
          const runtime: BrowserEditTaggerRuntime = {
            tokenizer,
            model,
            labels: labels.length ? labels : ['KEEP', 'SPACE_FIX', 'OPEN_REPLACE'],
            clsTokenId: specials.input_ids[0],
            sepTokenId: specials.input_ids[specials.input_ids.length - 1],
            maxLength: Number(config.max_position_embeddings ?? 512),
            source: candidate.source,
            root: candidate.root,
            dtype,
          };
          runtimeState = runtime;
          runtimeStatus = {
            ready: true,
            source: candidate.source,
            root: candidate.root,
            dtype,
          };
          return runtime;
        } catch (error) {
          lastError = error instanceof Error ? error.message : `Failed to load edit tagger (${dtype})`;
        }
      }
    } catch (error) {
      lastError = error instanceof Error ? error.message : `Failed to probe ${candidate.root}`;
    } finally {
      hfEnv.allowLocalModels = defaultAllowLocalModels;
      hfEnv.allowRemoteModels = defaultAllowRemoteModels;
      hfEnv.remoteHost = defaultRemoteHost;
      hfEnv.remotePathTemplate = defaultRemotePathTemplate;
    }
  }

  runtimeState = null;
  runtimeStatus = {
    ready: false,
    source: 'missing',
    error: lastError,
  };
  return null;
}

async function ensureRuntime(): Promise<BrowserEditTaggerRuntime | null> {
  if (runtimeState) return runtimeState;
  if (!runtimePromise) {
    runtimePromise = createRuntime().finally(() => {
      runtimePromise = null;
    });
  }
  return runtimePromise;
}

export function getBrowserEditTaggerStatus(): BrowserEditTaggerStatus {
  return runtimeStatus;
}

export async function warmupBrowserEditTagger(): Promise<BrowserEditTaggerStatus> {
  await ensureRuntime();
  return runtimeStatus;
}

export async function predictBrowserEditTagger(
  tokens: string[]
): Promise<BrowserEditTaggerPrediction[] | null> {
  if (!tokens.length) return [];

  const runtime = await ensureRuntime();
  if (!runtime) return null;

  const { tokenizer, model, clsTokenId, sepTokenId, maxLength, labels } = runtime;
  const sequenceIds: number[] = [clsTokenId];
  const tokenSubwordCounts: number[] = [];

  for (const token of tokens) {
    const encoded = (await tokenizer(token, {
      add_special_tokens: false,
      return_tensor: false,
    })) as { input_ids: number[] };

    const inputIds = encoded.input_ids ?? [];
    if (!inputIds.length) {
      tokenSubwordCounts.push(0);
      continue;
    }
    if (sequenceIds.length + inputIds.length + 1 > maxLength) {
      break;
    }
    tokenSubwordCounts.push(inputIds.length);
    sequenceIds.push(...inputIds);
  }
  sequenceIds.push(sepTokenId);

  const inputs = {
    input_ids: toBigIntTensor(sequenceIds),
    attention_mask: toBigIntTensor(new Array(sequenceIds.length).fill(1)),
    token_type_ids: toBigIntTensor(new Array(sequenceIds.length).fill(0)),
  };

  const outputs = await model(inputs);
  const logitsTensor = outputs.logits;
  const dims = logitsTensor.dims;
  const labelCount = dims[dims.length - 1];
  const raw = Array.from(logitsTensor.data as Float32Array);

  const predictions: BrowserEditTaggerPrediction[] = [];
  let cursor = 1;
  for (let tokenIndex = 0; tokenIndex < tokenSubwordCounts.length; tokenIndex += 1) {
    const subwordCount = tokenSubwordCounts[tokenIndex];
    if (subwordCount <= 0) {
      predictions.push({ label: 'KEEP', confidence: 0.5, logits: [0, 0, 0] });
      continue;
    }
    const averaged = new Array(labelCount).fill(0);
    for (let step = 0; step < subwordCount; step += 1) {
      for (let labelIndex = 0; labelIndex < labelCount; labelIndex += 1) {
        averaged[labelIndex] += raw[(cursor + step) * labelCount + labelIndex];
      }
    }
    for (let labelIndex = 0; labelIndex < labelCount; labelIndex += 1) {
      averaged[labelIndex] /= subwordCount;
    }
    const bestIndex = averaged.indexOf(Math.max(...averaged));
    predictions.push({
      label: (labels[bestIndex] ?? 'KEEP') as BrowserTaggerLabel,
      confidence: softmaxConfidence(averaged),
      logits: averaged,
    });
    cursor += subwordCount;
  }

  while (predictions.length < tokens.length) {
    predictions.push({ label: 'KEEP', confidence: 0.4, logits: [0, 0, 0] });
  }

  return predictions;
}
