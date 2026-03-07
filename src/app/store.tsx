import React, { createContext, useContext, useMemo, useReducer } from 'react';
import type { InputMode, PipelineResult, StageTrace } from '../pipeline/types';

export interface AppState {
  inputText: string;
  cursor: number;
  mode: InputMode;
  idleMs: number;
  provider: 'wasm' | 'webgpu';
  offlineSimulation: boolean;
  online: boolean;
  running: boolean;
  error: string | null;
  pipelineResult: PipelineResult | null;
  liveTrace: StageTrace[];
}

type Action =
  | { type: 'SET_INPUT'; value: string; cursor: number }
  | { type: 'SET_MODE'; value: InputMode }
  | { type: 'SET_IDLE'; value: number }
  | { type: 'SET_PROVIDER'; value: 'wasm' | 'webgpu' }
  | { type: 'SET_OFFLINE_SIM'; value: boolean }
  | { type: 'SET_ONLINE'; value: boolean }
  | { type: 'RUN_START' }
  | { type: 'RUN_ERROR'; message: string }
  | { type: 'RUN_DONE'; result: PipelineResult }
  | { type: 'SET_LIVE_TRACE'; traces: StageTrace[] };

const initialState: AppState = {
  inputText: '',
  cursor: 0,
  mode: 'inspect',
  idleMs: 900,
  provider: 'wasm',
  offlineSimulation: false,
  online: typeof navigator !== 'undefined' ? navigator.onLine : true,
  running: false,
  error: null,
  pipelineResult: null,
  liveTrace: [],
};

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_INPUT':
      return {
        ...state,
        inputText: action.value,
        cursor: action.cursor,
        error: null,
        pipelineResult: null,
        liveTrace: [],
      };
    case 'SET_MODE':
      return { ...state, mode: action.value };
    case 'SET_IDLE':
      return { ...state, idleMs: action.value };
    case 'SET_PROVIDER':
      return { ...state, provider: action.value };
    case 'SET_OFFLINE_SIM':
      return { ...state, offlineSimulation: action.value };
    case 'SET_ONLINE':
      return { ...state, online: action.value };
    case 'RUN_START':
      return { ...state, running: true, error: null, liveTrace: [] };
    case 'RUN_ERROR':
      return { ...state, running: false, error: action.message };
    case 'RUN_DONE':
      return { ...state, running: false, error: null, pipelineResult: action.result };
    case 'SET_LIVE_TRACE':
      return { ...state, liveTrace: action.traces };
    default:
      return state;
  }
}

const StoreContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<Action>;
} | null>(null);

export function AppStoreProvider({ children }: { children: React.ReactNode }): JSX.Element {
  const [state, dispatch] = useReducer(reducer, initialState);
  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useAppStore() {
  const ctx = useContext(StoreContext);
  if (!ctx) {
    throw new Error('useAppStore must be used inside AppStoreProvider');
  }
  return ctx;
}
