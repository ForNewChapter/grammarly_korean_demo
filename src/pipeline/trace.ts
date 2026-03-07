import type { StageTrace } from './types';

export function makeTrace<T>(
  stageName: string,
  inputText: string,
  output: T,
  startMs: number,
  endMs: number,
  why: string
): StageTrace {
  return {
    stageName,
    inputText,
    output,
    latencyMs: Math.max(0, Number((endMs - startMs).toFixed(2))),
    why,
  };
}
