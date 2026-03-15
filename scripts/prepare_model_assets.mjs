import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

const expectedModels = [
  'profile_classifier.onnx',
  'spacing_boundary.onnx',
  'edit_tagger.onnx',
  'reranker.onnx',
  'guardrail.onnx',
];

const sourceDir = path.resolve(
  projectRoot,
  process.env.LOCAL_MODEL_SOURCE_DIR ?? path.join('.local-assets', 'models')
);
const targetDir = path.resolve(
  projectRoot,
  process.env.LOCAL_MODEL_TARGET_DIR ?? path.join('public', 'assets', 'models-local')
);
const strict = process.env.LOCAL_MODELS_REQUIRED === '1';
const minModelBytes = 1024;

async function pathExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function ensureDir(targetPath) {
  await fs.mkdir(targetPath, { recursive: true });
}

async function removeIfExists(targetPath) {
  try {
    await fs.rm(targetPath, { force: true });
  } catch {
    // ignore
  }
}

async function syncModel(fileName) {
  const sourcePath = path.join(sourceDir, fileName);
  const targetPath = path.join(targetDir, fileName);

  if (!(await pathExists(sourcePath))) {
    await removeIfExists(targetPath);
    return { fileName, status: 'missing' };
  }

  const stat = await fs.stat(sourcePath);
  if (stat.size < minModelBytes) {
    await removeIfExists(targetPath);
    return { fileName, status: 'placeholder', size: stat.size };
  }

  await fs.copyFile(sourcePath, targetPath);
  return { fileName, status: 'ready', size: stat.size };
}

async function main() {
  await ensureDir(targetDir);

  const results = [];
  for (const fileName of expectedModels) {
    results.push(await syncModel(fileName));
  }

  const ready = results.filter((result) => result.status === 'ready');
  const missing = results.filter((result) => result.status === 'missing');
  const placeholders = results.filter((result) => result.status === 'placeholder');

  for (const result of results) {
    if (result.status === 'ready') {
      console.log(`model asset ready: ${result.fileName} (${result.size} bytes)`);
    } else if (result.status === 'placeholder') {
      console.warn(`model asset skipped: ${result.fileName} (${result.size} bytes, too small)`);
    } else {
      console.warn(`model asset missing: ${result.fileName}`);
    }
  }

  if (ready.length === 0) {
    const message =
      `No local runtime models were prepared from ${sourceDir}. ` +
      `The browser will fall back to heuristic mode and ignore bundled placeholder ONNX files.`;
    if (strict) {
      throw new Error(message);
    }
    console.warn(message);
  }

  if (strict && (missing.length > 0 || placeholders.length > 0)) {
    throw new Error(
      `LOCAL_MODELS_REQUIRED=1 but ${missing.length + placeholders.length} runtime model(s) are unavailable.`
    );
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
