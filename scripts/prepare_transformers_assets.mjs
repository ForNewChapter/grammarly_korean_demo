import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

const sourceFiles = [
  {
    from: path.join(
      projectRoot,
      'node_modules',
      '@huggingface',
      'transformers',
      'dist',
      'ort-wasm-simd-threaded.jsep.wasm'
    ),
    to: path.join(projectRoot, 'public', 'assets', 'transformers', 'ort-wasm-simd-threaded.jsep.wasm'),
  },
];

async function ensureDir(targetPath) {
  await fs.mkdir(path.dirname(targetPath), { recursive: true });
}

async function main() {
  for (const file of sourceFiles) {
    await ensureDir(file.to);
    await fs.copyFile(file.from, file.to);
    const stat = await fs.stat(file.to);
    console.log(`transformers asset ready: ${path.basename(file.to)} (${stat.size} bytes)`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
