import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

const kiwiDir = path.join(projectRoot, 'public', 'assets', 'kiwi', 'model');
const localModelDir = path.join(projectRoot, 'public', 'assets', 'models-local');
const transformerDir = path.join(projectRoot, 'public', 'assets', 'transformers');
const browserManifestPath = path.join(projectRoot, 'public', 'assets', 'browser_model_manifest.json');
const browserEditTaggerDirs = [
  path.join(projectRoot, 'public', 'assets', 'browser-models-local', 'edit_tagger_v2'),
  path.join(projectRoot, 'public', 'assets', 'browser-models', 'edit_tagger_v2'),
];

const kiwiFiles = [
  'combiningRule.txt',
  'cong.mdl',
  'default.dict',
  'dialect.dict',
  'extract.mdl',
  'multi.dict',
  'nounchr.mdl',
  'sj.morph',
  'typo.dict',
];

const modelFiles = [
  'profile_classifier.onnx',
  'spacing_boundary.onnx',
  'edit_tagger.onnx',
  'reranker.onnx',
  'guardrail.onnx',
];

const strictModels = process.argv.includes('--strict-models');
const minModelBytes = 1024;

async function statOrNull(targetPath) {
  try {
    return await fs.stat(targetPath);
  } catch {
    return null;
  }
}

async function main() {
  let failed = false;

  console.log(`validate: kiwi dir = ${kiwiDir}`);
  for (const fileName of kiwiFiles) {
    const stat = await statOrNull(path.join(kiwiDir, fileName));
    if (!stat || stat.size <= 0) {
      console.error(`missing kiwi asset: ${fileName}`);
      failed = true;
    } else {
      console.log(`kiwi ok: ${fileName} (${stat.size} bytes)`);
    }
  }

  console.log(`validate: local model dir = ${localModelDir}`);
  let readyModels = 0;
  for (const fileName of modelFiles) {
    const stat = await statOrNull(path.join(localModelDir, fileName));
    if (!stat) {
      console.warn(`local model missing: ${fileName}`);
      continue;
    }
    if (stat.size < minModelBytes) {
      console.warn(`local model too small: ${fileName} (${stat.size} bytes)`);
      continue;
    }
    readyModels += 1;
    console.log(`local model ok: ${fileName} (${stat.size} bytes)`);
  }

  if (strictModels && readyModels !== modelFiles.length) {
    console.error(
      `strict model validation failed: expected ${modelFiles.length} real model files, found ${readyModels}`
    );
    failed = true;
  } else if (readyModels === 0) {
    console.warn('no real local browser models found; browser runtime will use heuristic path');
  }

  const transformerStat = await statOrNull(path.join(transformerDir, 'ort-wasm-simd-threaded.jsep.wasm'));
  if (!transformerStat || transformerStat.size <= 0) {
    console.error('missing transformers runtime asset: ort-wasm-simd-threaded.jsep.wasm');
    failed = true;
  } else {
    console.log(`transformers ok: ort-wasm-simd-threaded.jsep.wasm (${transformerStat.size} bytes)`);
  }

  const manifestStat = await statOrNull(browserManifestPath);
  if (!manifestStat || manifestStat.size <= 0) {
    console.error('missing browser model manifest: public/assets/browser_model_manifest.json');
    failed = true;
  } else {
    console.log(`browser model manifest ok: ${manifestStat.size} bytes`);
  }

  for (const browserDir of browserEditTaggerDirs) {
    const configStat = await statOrNull(path.join(browserDir, 'config.json'));
    const tokenizerStat = await statOrNull(path.join(browserDir, 'tokenizer.json'));
    const onnxStat = await statOrNull(path.join(browserDir, 'onnx', 'model.onnx'));
    if (configStat && tokenizerStat && onnxStat && onnxStat.size > minModelBytes) {
      console.log(`browser edit tagger ok: ${browserDir} (onnx=${onnxStat.size} bytes)`);
    } else {
      console.warn(`browser edit tagger missing/incomplete: ${browserDir}`);
    }
  }

  if (failed) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
