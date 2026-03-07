import type { InputMode, PipelineResult, StageTrace } from '../pipeline/types';

export type WorkerRequest =
  | { type: 'INIT'; provider: 'wasm' | 'webgpu' }
  | { type: 'RUN_PIPELINE'; text: string; cursor: number; mode: InputMode }
  | { type: 'RUN_STAGE'; stage: string; payload: unknown }
  | { type: 'WARMUP'; models: string[] };

export type WorkerResponse =
  | {
      type: 'INIT_DONE';
      provider: 'wasm' | 'webgpu';
      modelReady: Record<string, boolean>;
      assetStatus: {
        appShellReady: boolean;
        rulesReady: boolean;
        smallModelsReady: boolean;
        offlineCapable: boolean;
      };
    }
  | { type: 'PIPELINE_RESULT'; result: PipelineResult }
  | { type: 'STAGE_RESULT'; stage: string; trace: StageTrace }
  | { type: 'ERROR'; message: string };
