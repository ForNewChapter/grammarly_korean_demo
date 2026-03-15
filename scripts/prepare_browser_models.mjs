import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const args = {
    target: 'local',
    dtype: 'fp32',
    sourceDir: path.join(projectRoot, 'models', 'edit_tagger_v2_best', 'best'),
    outputDir: '',
    force: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--target' && argv[i + 1]) {
      args.target = argv[++i];
    } else if (arg === '--dtype' && argv[i + 1]) {
      args.dtype = argv[++i];
    } else if (arg === '--source-dir' && argv[i + 1]) {
      args.sourceDir = path.resolve(projectRoot, argv[++i]);
    } else if (arg === '--output-dir' && argv[i + 1]) {
      args.outputDir = path.resolve(projectRoot, argv[++i]);
    } else if (arg === '--force') {
      args.force = true;
    }
  }

  if (!args.outputDir) {
    const baseDir = args.target === 'bundled' ? 'browser-models' : 'browser-models-local';
    args.outputDir = path.join(projectRoot, 'public', 'assets', baseDir, 'edit_tagger_v2');
  }

  return args;
}

async function exists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function statOrNull(targetPath) {
  try {
    return await fs.stat(targetPath);
  } catch {
    return null;
  }
}

async function ensureCleanDir(targetDir) {
  await fs.rm(targetDir, { recursive: true, force: true });
  await fs.mkdir(targetDir, { recursive: true });
}

async function resolvePythonExecutable() {
  const candidates = [
    process.env.BROWSER_EXPORT_PYTHON,
    path.join(projectRoot, '.venv', 'bin', 'python'),
    'python3',
  ].filter(Boolean);

  for (const candidate of candidates) {
    try {
      await runCommand(candidate, ['--version'], { stdio: 'ignore' });
      return candidate;
    } catch {
      // try next
    }
  }
  throw new Error('no usable python interpreter found for browser model export');
}

async function isUpToDate(sourceDir, outputDir) {
  const outputFiles = [
    path.join(outputDir, 'config.json'),
    path.join(outputDir, 'tokenizer.json'),
    path.join(outputDir, 'tokenizer_config.json'),
    path.join(outputDir, 'onnx', 'model.onnx'),
  ];
  const inputFiles = [
    path.join(sourceDir, 'config.json'),
    path.join(sourceDir, 'tokenizer.json'),
    path.join(sourceDir, 'tokenizer_config.json'),
    path.join(sourceDir, 'model.safetensors'),
  ];

  const outputStats = await Promise.all(outputFiles.map((file) => statOrNull(file)));
  if (outputStats.some((stat) => !stat || stat.size <= 0)) {
    return false;
  }
  const inputStats = await Promise.all(inputFiles.map((file) => statOrNull(file)));
  if (inputStats.some((stat) => !stat)) {
    return false;
  }

  const latestInput = Math.max(...inputStats.map((stat) => stat.mtimeMs));
  const earliestOutput = Math.min(...outputStats.map((stat) => stat.mtimeMs));
  return earliestOutput >= latestInput;
}

function runCommand(cmd, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      stdio: 'inherit',
      cwd: projectRoot,
      ...options,
    });
    child.on('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${cmd} ${args.join(' ')} exited with code ${code}`));
      }
    });
    child.on('error', reject);
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const exporter = path.join(projectRoot, 'scripts', 'export_browser_edit_tagger.py');

  if (!(await exists(args.sourceDir))) {
    throw new Error(`source model dir not found: ${args.sourceDir}`);
  }
  if (!(await exists(exporter))) {
    throw new Error(`export script not found: ${exporter}`);
  }

  if (!args.force && (await isUpToDate(args.sourceDir, args.outputDir))) {
    const onnxStat = await fs.stat(path.join(args.outputDir, 'onnx', 'model.onnx'));
    console.log(`browser edit tagger already up to date: ${args.outputDir} (${onnxStat.size} bytes)`);
    return;
  }

  const python = await resolvePythonExecutable();
  await ensureCleanDir(args.outputDir);

  await runCommand(python, [
    exporter,
    '--input-dir',
    args.sourceDir,
    '--output-dir',
    args.outputDir,
    '--dtype',
    args.dtype,
  ]);

  const configStat = await fs.stat(path.join(args.outputDir, 'config.json'));
  const tokenizerStat = await fs.stat(path.join(args.outputDir, 'tokenizer.json'));
  const onnxStat = await fs.stat(path.join(args.outputDir, 'onnx', 'model.onnx'));
  console.log(`browser edit tagger ready: config=${configStat.size} tokenizer=${tokenizerStat.size} onnx=${onnxStat.size}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
