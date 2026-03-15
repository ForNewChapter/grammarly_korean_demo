import * as ort from 'onnxruntime-web';

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0.5;
  return Math.max(0, Math.min(1, v));
}

function sigmoid(x: number): number {
  if (!Number.isFinite(x)) return 0.5;
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function inferTensorType(metaType: string | undefined): ort.Tensor.Type {
  const t = (metaType ?? '').toLowerCase();
  if (t.includes('int64')) return 'int64';
  if (t.includes('int32')) return 'int32';
  if (t.includes('bool')) return 'bool';
  return 'float32';
}

function normalizeDims(rawDims: ReadonlyArray<number | string> | undefined, fallbackSize: number): number[] {
  if (!rawDims || rawDims.length === 0) return [1, fallbackSize];
  const dims = rawDims.map((d) => (typeof d === 'number' && d > 0 ? d : 1));
  if (dims.reduce((acc, v) => acc * v, 1) <= 0) return [1, fallbackSize];
  return dims;
}

function toTensorData(type: ort.Tensor.Type, values: number[], size: number): ArrayLike<number | bigint | boolean> {
  const source = values.length ? values : [0];

  if (type === 'int64') {
    const out = new BigInt64Array(size);
    for (let i = 0; i < size; i += 1) {
      out[i] = BigInt(Math.round(source[i % source.length]));
    }
    return out;
  }

  if (type === 'int32') {
    const out = new Int32Array(size);
    for (let i = 0; i < size; i += 1) {
      out[i] = Math.round(source[i % source.length]);
    }
    return out;
  }

  if (type === 'bool') {
    const out = new Uint8Array(size);
    for (let i = 0; i < size; i += 1) {
      out[i] = source[i % source.length] > 0 ? 1 : 0;
    }
    return out;
  }

  const out = new Float32Array(size);
  for (let i = 0; i < size; i += 1) {
    out[i] = source[i % source.length];
  }
  return out;
}

export function makeTextFeatures(text: string): number[] {
  const len = Math.max(1, text.length);
  const hangul = (text.match(/[가-힣]/g) ?? []).length / len;
  const latin = (text.match(/[A-Za-z]/g) ?? []).length / len;
  const digits = (text.match(/\d/g) ?? []).length / len;
  const spaces = (text.match(/\s/g) ?? []).length / len;
  const punct = (text.match(/[!?.,]/g) ?? []).length / len;
  const chat = /[ㅋㅎㅠ]{2,}|[!?]{2,}|\bㄹㅇ\b|\b개좋/.test(text) ? 1 : 0;
  return [len / 256, hangul, latin, digits, spaces, punct, chat, 1 - hangul];
}

export function makeTokenFeatures(token: string): number[] {
  const len = Math.max(1, token.length);
  const hangul = (token.match(/[가-힣]/g) ?? []).length / len;
  const hasDigit = /\d/.test(token) ? 1 : 0;
  const hasLatin = /[A-Za-z]/.test(token) ? 1 : 0;
  const suspicious = /낫|되요|할수|됬|않되|안되/.test(token) ? 1 : 0;
  const jamoNoise = /[ㄱ-ㅎㅏ-ㅣ]/.test(token) ? 1 : 0;
  return [len / 32, hangul, hasDigit, hasLatin, suspicious, jamoNoise, token.endsWith('요') ? 1 : 0, token.endsWith('다') ? 1 : 0];
}

export function makeBoundaryFeatures(
  text: string,
  index: number,
  baseScore: number,
  profileCode: number
): number[] {
  const left = text.slice(Math.max(0, index - 3), index);
  const right = text.slice(index, Math.min(text.length, index + 3));
  const leftHangul = (left.match(/[가-힣]/g) ?? []).length;
  const rightHangul = (right.match(/[가-힣]/g) ?? []).length;
  const leftPunct = /[!?.,]/.test(left) ? 1 : 0;
  const rightPunct = /[!?.,]/.test(right) ? 1 : 0;
  return [baseScore, profileCode, leftHangul / 3, rightHangul / 3, leftPunct, rightPunct, index / Math.max(1, text.length), 1];
}

export function makeRerankFeatures(
  text: string,
  original: string,
  replacement: string,
  generatorScore: number,
  baseScore: number,
  profileCode: number
): number[] {
  const sentence = text.replace(original, replacement);
  const deltaLen = (replacement.length - original.length) / Math.max(1, original.length);
  const hasHangul = /[가-힣]/.test(sentence) ? 1 : 0;
  const hasAlphaNum = /[A-Za-z0-9]/.test(replacement) ? 1 : 0;
  return [generatorScore, baseScore, profileCode, deltaLen, hasHangul, hasAlphaNum, replacement === original ? 1 : 0, sentence.length / 256];
}

function readFirstNumericVector(result: ort.InferenceSession.ReturnType): number[] | null {
  for (const value of Object.values(result)) {
    const candidate = value as { data?: ArrayLike<number | bigint> };
    if (!candidate || !candidate.data) continue;
    const raw = Array.from(candidate.data, (v) => (typeof v === 'bigint' ? Number(v) : Number(v)));
    const nums = raw.filter((v) => Number.isFinite(v));
    if (nums.length > 0) return nums;
  }
  return null;
}

export async function runNumericModel(
  session: ort.InferenceSession,
  features: number[]
): Promise<number[] | null> {
  const feeds: Record<string, ort.Tensor> = {};

  for (let i = 0; i < session.inputNames.length; i += 1) {
    const name = session.inputNames[i];
    const meta = session.inputMetadata[i];
    const type = meta && meta.isTensor ? inferTensorType(meta.type) : inferTensorType(undefined);
    const dims = meta && meta.isTensor ? normalizeDims(meta.shape, features.length || 8) : normalizeDims(undefined, features.length || 8);
    const size = dims.reduce((acc, v) => acc * v, 1);
    const data = toTensorData(type, features, size);
    feeds[name] = new ort.Tensor(type, data as never, dims);
  }

  try {
    const result = await session.run(feeds);
    return readFirstNumericVector(result);
  } catch {
    return null;
  }
}

export function scoreFromVector(vec: number[] | null, fallback = 0.5): number {
  if (!vec || vec.length === 0) return fallback;
  if (vec.length === 1) return clamp01(sigmoid(vec[0]));

  const max = Math.max(...vec);
  const min = Math.min(...vec);
  if (!Number.isFinite(max) || !Number.isFinite(min)) return fallback;
  if (vec.length >= 2) {
    const logit = vec[1] - vec[0];
    return clamp01(sigmoid(logit));
  }
  return clamp01((max - min) / (Math.abs(max) + Math.abs(min) + 1e-6));
}

export function argmax(values: number[]): { index: number; value: number } {
  let bestIdx = 0;
  let best = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > best) {
      best = values[i];
      bestIdx = i;
    }
  }
  return { index: bestIdx, value: best };
}

export function softmaxConfidence(values: number[]): number {
  if (values.length === 0) return 0;
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (sum <= 0) return 0;
  return Math.max(...exps) / sum;
}

export function blend(base: number, model: number, modelWeight = 0.35): number {
  const w = clamp01(modelWeight);
  return clamp01(base * (1 - w) + model * w);
}
