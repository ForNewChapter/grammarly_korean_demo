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
      modelSource?: Record<string, 'local' | 'bundled' | 'remote' | 'missing'>;
      assetStatus: {
        appShellReady: boolean;
        rulesReady: boolean;
        kiwiReady?: boolean;
        smallModelsReady: boolean;
        offlineCapable: boolean;
        runtimeMode?: 'browser' | 'backend';
        modelSource?: Record<string, 'local' | 'bundled' | 'remote' | 'backend' | 'missing'>;
        assetErrors?: string[];
      };
    }
  | {
      type: 'WARMUP_DONE';
      provider: 'wasm' | 'webgpu';
      modelReady: Record<string, boolean>;
      modelSource?: Record<string, 'local' | 'bundled' | 'remote' | 'missing'>;
      assetStatus: {
        appShellReady: boolean;
        rulesReady: boolean;
        kiwiReady?: boolean;
        smallModelsReady: boolean;
        offlineCapable: boolean;
        runtimeMode?: 'browser' | 'backend';
        modelSource?: Record<string, 'local' | 'bundled' | 'remote' | 'backend' | 'missing'>;
        assetErrors?: string[];
      };
    }
  | { type: 'PIPELINE_RESULT'; result: PipelineResult }
  | { type: 'STAGE_RESULT'; stage: string; trace: StageTrace }
  | { type: 'ERROR'; message: string };
