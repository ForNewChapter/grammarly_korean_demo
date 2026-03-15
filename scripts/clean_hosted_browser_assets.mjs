import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

const targets = [
  path.join(projectRoot, 'public', 'assets', 'browser-models-local'),
  path.join(projectRoot, 'public', 'assets', 'browser-models'),
];

async function main() {
  for (const target of targets) {
    await fs.rm(target, { recursive: true, force: true });
    console.log(`removed hosted-excluded asset dir: ${target}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
