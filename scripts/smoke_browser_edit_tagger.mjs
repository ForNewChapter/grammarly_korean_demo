import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  AutoModelForTokenClassification,
  AutoTokenizer,
  Tensor,
  env as hfEnv,
} from '@huggingface/transformers';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const args = {
    modelDir: path.join(projectRoot, 'public', 'assets', 'browser-models-local', 'edit_tagger_v2'),
    tokens: ['아기를', '나았다'],
    expect: ['KEEP', 'OPEN_REPLACE'],
    dtype: 'fp32',
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model-dir' && argv[i + 1]) {
      const value = argv[++i];
      args.modelDir = /^https?:\/\//.test(value) ? value : path.resolve(projectRoot, value);
    } else if (arg === '--tokens' && argv[i + 1]) {
      args.tokens = argv[++i].split(',').map((item) => item.trim()).filter(Boolean);
    } else if (arg === '--expect' && argv[i + 1]) {
      args.expect = argv[++i].split(',').map((item) => item.trim()).filter(Boolean);
    } else if (arg === '--dtype' && argv[i + 1]) {
      args.dtype = argv[++i];
    }
  }
  return args;
}

function toBigIntTensor(values) {
  return new Tensor('int64', BigInt64Array.from(values.map((value) => BigInt(value))), [1, values.length]);
}

function parseRemoteRoot(root) {
  const url = new URL(root);
  const modelId = url.pathname.replace(/^\/+/, '').replace(/\/+$/, '');
  if (!modelId) {
    throw new Error(`Remote model root is missing a model id: ${root}`);
  }
  return {
    modelId,
    remoteHost: `${url.origin}/`,
    remotePathTemplate: '{model}/',
  };
}

function averageSubwordLogits(raw, labelCount, cursor, subwordCount) {
  const averaged = new Array(labelCount).fill(0);
  for (let step = 0; step < subwordCount; step += 1) {
    for (let labelIndex = 0; labelIndex < labelCount; labelIndex += 1) {
      averaged[labelIndex] += raw[(cursor + step) * labelCount + labelIndex];
    }
  }
  for (let labelIndex = 0; labelIndex < labelCount; labelIndex += 1) {
    averaged[labelIndex] /= subwordCount;
  }
  return averaged;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const isRemoteModel = /^https?:\/\//.test(args.modelDir);

  const defaultAllowLocalModels = hfEnv.allowLocalModels;
  const defaultAllowRemoteModels = hfEnv.allowRemoteModels;
  const defaultRemoteHost = hfEnv.remoteHost;
  const defaultRemotePathTemplate = hfEnv.remotePathTemplate;

  let modelRef = args.modelDir;
  if (isRemoteModel) {
    const remoteConfig = parseRemoteRoot(args.modelDir);
    modelRef = remoteConfig.modelId;
    hfEnv.allowLocalModels = false;
    hfEnv.allowRemoteModels = true;
    hfEnv.remoteHost = remoteConfig.remoteHost;
    hfEnv.remotePathTemplate = remoteConfig.remotePathTemplate;
  } else {
    hfEnv.allowLocalModels = true;
    hfEnv.allowRemoteModels = false;
  }

  try {
    const tokenizer = await AutoTokenizer.from_pretrained(modelRef, {
      local_files_only: !isRemoteModel,
    });
    const model = await AutoModelForTokenClassification.from_pretrained(modelRef, {
      local_files_only: !isRemoteModel,
      dtype: args.dtype,
    });

    const config = model.config;
    const id2label = config.id2label ?? {};
    const labels = Object.values(id2label);
    const specials = await tokenizer('', { return_tensor: false });
    const clsTokenId = specials.input_ids[0];
    const sepTokenId = specials.input_ids[specials.input_ids.length - 1];
    const maxLength = Number(config.max_position_embeddings ?? 512);

    const sequenceIds = [clsTokenId];
    const tokenSubwordCounts = [];
    for (const token of args.tokens) {
      const encoded = await tokenizer(token, { add_special_tokens: false, return_tensor: false });
      const inputIds = encoded.input_ids ?? [];
      if (!inputIds.length) {
        tokenSubwordCounts.push(0);
        continue;
      }
      if (sequenceIds.length + inputIds.length + 1 > maxLength) {
        throw new Error(`token sequence exceeds model max length: ${token}`);
      }
      tokenSubwordCounts.push(inputIds.length);
      sequenceIds.push(...inputIds);
    }
    sequenceIds.push(sepTokenId);

    const outputs = await model({
      input_ids: toBigIntTensor(sequenceIds),
      attention_mask: toBigIntTensor(new Array(sequenceIds.length).fill(1)),
      token_type_ids: toBigIntTensor(new Array(sequenceIds.length).fill(0)),
    });

    const logitsTensor = outputs.logits;
    const raw = Array.from(logitsTensor.data);
    const labelCount = logitsTensor.dims[logitsTensor.dims.length - 1];

    let cursor = 1;
    const predictions = [];
    for (let tokenIndex = 0; tokenIndex < tokenSubwordCounts.length; tokenIndex += 1) {
      const subwordCount = tokenSubwordCounts[tokenIndex];
      if (subwordCount <= 0) {
        predictions.push('KEEP');
        continue;
      }
      const averaged = averageSubwordLogits(raw, labelCount, cursor, subwordCount);
      const bestIndex = averaged.indexOf(Math.max(...averaged));
      predictions.push(labels[bestIndex] ?? 'KEEP');
      cursor += subwordCount;
    }

    console.log(`tokens=${args.tokens.join(' | ')}`);
    console.log(`predictions=${predictions.join(' | ')}`);
    if (args.expect.length) {
      console.log(`expected=${args.expect.join(' | ')}`);
      const ok =
        args.expect.length === predictions.length &&
        args.expect.every((label, index) => label === predictions[index]);
      if (!ok) {
        throw new Error(`browser edit tagger smoke test failed: expected ${args.expect.join(',')} got ${predictions.join(',')}`);
      }
    }
  } finally {
    hfEnv.allowLocalModels = defaultAllowLocalModels;
    hfEnv.allowRemoteModels = defaultAllowRemoteModels;
    hfEnv.remoteHost = defaultRemoteHost;
    hfEnv.remotePathTemplate = defaultRemotePathTemplate;
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
