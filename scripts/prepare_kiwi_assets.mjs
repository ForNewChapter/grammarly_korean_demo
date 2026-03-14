import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const targetDir = path.join(projectRoot, 'public', 'assets', 'kiwi', 'model');
const rawBaseUrl = 'https://raw.githubusercontent.com/bab2min/Kiwi/main/models/cong/base';
const mediaBaseUrl = 'https://media.githubusercontent.com/media/bab2min/Kiwi/main/models/cong/base';

const modelFiles = {
  'combiningRule.txt': `${rawBaseUrl}/combiningRule.txt`,
  'cong.mdl': `${mediaBaseUrl}/cong.mdl`,
  'default.dict': `${rawBaseUrl}/default.dict`,
  'dialect.dict': `${rawBaseUrl}/dialect.dict`,
  'extract.mdl': `${mediaBaseUrl}/extract.mdl`,
  'multi.dict': `${rawBaseUrl}/multi.dict`,
  'nounchr.mdl': `${mediaBaseUrl}/nounchr.mdl`,
  'sj.morph': `${mediaBaseUrl}/sj.morph`,
  'typo.dict': `${rawBaseUrl}/typo.dict`,
};

async function ensureFile(fileName) {
  const targetPath = path.join(targetDir, fileName);
  if (process.env.KIWI_REFRESH !== '1') {
    try {
      const stat = await fs.stat(targetPath);
      if (stat.size > 0) {
        return;
      }
    } catch {
      // download below
    }
  }

  const url = modelFiles[fileName];
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download Kiwi model asset: ${fileName} <- ${url} (${res.status})`);
  }

  const data = Buffer.from(await res.arrayBuffer());
  const tempPath = `${targetPath}.tmp`;
  await fs.writeFile(tempPath, data);
  await fs.rename(tempPath, targetPath);
  console.log(`kiwi asset ready: ${fileName} (${data.length} bytes)`);
}

async function main() {
  if (process.env.KIWI_SKIP_DOWNLOAD === '1') {
    console.log('Skipping Kiwi asset download because KIWI_SKIP_DOWNLOAD=1');
    return;
  }

  await fs.mkdir(targetDir, { recursive: true });
  for (const fileName of Object.keys(modelFiles)) {
    await ensureFile(fileName);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
