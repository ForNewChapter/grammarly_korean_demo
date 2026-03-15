import { useCallback, useEffect, useMemo, useRef } from 'react';
import { EditorPanel } from '../components/EditorPanel';
import { FinalOutputPanel } from '../components/FinalOutputPanel';
import { AssetStatusPanel } from '../components/AssetStatusPanel';
import { PipelineOrchestrator } from '../pipeline/orchestrator';
import { useAppStore } from './store';
import { getSetting, saveTraceSession } from '../cache/indexedDb';
import { applyEditsWithPreview } from '../utils/editPreview';

export default function App(): JSX.Element {
  const { state, dispatch } = useAppStore();
  const orchestratorRef = useRef<PipelineOrchestrator | null>(null);
  const latestStateRef = useRef(state);

  useEffect(() => {
    latestStateRef.current = state;
  }, [state]);

  const modelReady = useMemo(
    () => state.pipelineResult?.assetStatus.modelReady ?? state.runtimeStatus?.modelReady ?? {},
    [state.pipelineResult?.assetStatus.modelReady, state.runtimeStatus?.modelReady]
  );
  const modelSource = useMemo(
    () => state.pipelineResult?.assetStatus.modelSource ?? state.runtimeStatus?.assetStatus.modelSource ?? {},
    [state.pipelineResult?.assetStatus.modelSource, state.runtimeStatus?.assetStatus.modelSource]
  );
  const runtimeMode =
    state.pipelineResult?.assetStatus.runtimeMode ?? state.runtimeStatus?.assetStatus.runtimeMode ?? 'browser';
  const rulesReady = state.pipelineResult?.assetStatus.rulesReady ?? state.runtimeStatus?.assetStatus.rulesReady ?? false;
  const kiwiReady = state.pipelineResult?.assetStatus.kiwiReady ?? state.runtimeStatus?.assetStatus.kiwiReady ?? false;
  const assetErrors = state.pipelineResult?.assetStatus.assetErrors ?? state.runtimeStatus?.assetStatus.assetErrors ?? [];
  const backendStatus =
    state.pipelineResult?.assetStatus.backend ?? state.runtimeStatus?.assetStatus.backend ?? undefined;

  const runPipeline = useCallback(async () => {
    if (!orchestratorRef.current) return;
    const currentState = latestStateRef.current;
    dispatch({ type: 'RUN_START' });
    try {
      const result = await orchestratorRef.current.runPipeline(
        currentState.inputText,
        currentState.cursor,
        currentState.mode
      );
      dispatch({ type: 'RUN_DONE', result });
      await saveTraceSession(result.original, result.corrected, result.traces);
    } catch (error) {
      dispatch({
        type: 'RUN_ERROR',
        message: error instanceof Error ? error.message : '파이프라인 실행 실패',
      });
    }
  }, [dispatch]);

  useEffect(() => {
    const onOnline = () => dispatch({ type: 'SET_ONLINE', value: true });
    const onOffline = () => dispatch({ type: 'SET_ONLINE', value: false });
    window.addEventListener('online', onOnline);
    window.addEventListener('offline', onOffline);

    const orchestrator = new PipelineOrchestrator();
    orchestratorRef.current = orchestrator;

    (async () => {
      const savedProvider = await getSetting<'wasm' | 'webgpu'>('provider', 'wasm');
      dispatch({ type: 'SET_PROVIDER', value: savedProvider });
      dispatch({ type: 'SET_MODE', value: 'inspect' });

      const initResult = await orchestrator.init(savedProvider);
      dispatch({ type: 'SET_RUNTIME_STATUS', status: initResult });
      const warmupResult = await orchestrator.warmup([
        'profile_classifier',
        'spacing_boundary',
        'edit_tagger',
        'reranker',
        'guardrail',
      ]);
      if (warmupResult) {
        dispatch({ type: 'SET_RUNTIME_STATUS', status: warmupResult });
      }
    })().catch((e: unknown) => {
      dispatch({ type: 'RUN_ERROR', message: e instanceof Error ? e.message : '초기화 실패' });
    });

    return () => {
      window.removeEventListener('online', onOnline);
      window.removeEventListener('offline', onOffline);
      orchestrator.dispose();
    };
  }, [dispatch, runPipeline]);

  const corrected = state.pipelineResult?.corrected ?? '';
  const suggestedText = state.pipelineResult?.suggestedText ?? '';
  const baseForProposal = corrected || state.pipelineResult?.original || state.inputText;
  const suggestEdits = useMemo(
    () => (state.pipelineResult?.edits ?? []).filter((edit) => !edit.autoApplicable && edit.editType === 'OPEN_REPLACE'),
    [state.pipelineResult?.edits]
  );
  const localSuggestionPreview = useMemo(
    () => applyEditsWithPreview(baseForProposal, suggestEdits, 'suggestion'),
    [baseForProposal, suggestEdits]
  );
  const proposalText = suggestedText || localSuggestionPreview.text || baseForProposal;
  const proposalPreview = useMemo(
    () =>
      suggestedText && suggestedText === localSuggestionPreview.text
        ? localSuggestionPreview
        : applyEditsWithPreview(proposalText, [], 'suggestion'),
    [localSuggestionPreview, proposalText, suggestedText]
  );

  return (
    <div className="app-root">
      <header className="topbar">
        <div>
          <h1>한국어 맞춤법 교정 테스트 페이지</h1>
          <p>문장을 입력하고 현재 연결된 백엔드의 교정 결과를 확인합니다.</p>
        </div>
        <div>
          <AssetStatusPanel
            provider={state.provider}
            modelReady={modelReady}
            runtimeMode={runtimeMode}
            modelSource={modelSource}
            rulesReady={rulesReady}
            kiwiReady={kiwiReady}
            assetErrors={assetErrors}
            backend={backendStatus}
          />
        </div>
      </header>

      {state.error ? <div className="error-banner">오류: {state.error}</div> : null}

      <main className="layout layout-simple">
        <div className="left-col">
          <EditorPanel
            inputText={state.inputText}
            running={state.running}
            runtimeMode={runtimeMode}
            onChangeText={(value, cursor) => dispatch({ type: 'SET_INPUT', value, cursor })}
            onRun={() => void runPipeline()}
          />

          <FinalOutputPanel
            currentText={state.inputText}
            proposalText={proposalText}
            proposalMarkers={proposalPreview.markers}
            onApply={() => dispatch({ type: 'SET_INPUT', value: proposalText, cursor: proposalText.length })}
          />
        </div>
      </main>
    </div>
  );
}
