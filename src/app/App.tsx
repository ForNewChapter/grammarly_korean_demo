import { useCallback, useEffect, useMemo, useRef } from 'react';
import { registerSW } from 'virtual:pwa-register';
import { EditorPanel } from '../components/EditorPanel';
import { TracePanel } from '../components/TracePanel';
import { CandidatePanel } from '../components/CandidatePanel';
import { FinalOutputPanel } from '../components/FinalOutputPanel';
import { OfflineBadge } from '../components/OfflineBadge';
import { AssetStatusPanel } from '../components/AssetStatusPanel';
import { PipelineOrchestrator } from '../pipeline/orchestrator';
import { useAppStore } from './store';
import { getSetting, saveTraceSession, setSetting } from '../cache/indexedDb';
import { applyEditsWithPreview } from '../utils/editPreview';

export default function App(): JSX.Element {
  const { state, dispatch } = useAppStore();
  const orchestratorRef = useRef<PipelineOrchestrator | null>(null);
  const latestStateRef = useRef(state);

  useEffect(() => {
    latestStateRef.current = state;
  }, [state]);

  const modelReady = useMemo(
    () => state.pipelineResult?.assetStatus.modelReady ?? {},
    [state.pipelineResult?.assetStatus.modelReady]
  );

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
    const pwaUpdate = registerSW({
      onNeedRefresh() {
        console.log('새 버전 업데이트 가능');
      },
      onOfflineReady() {
        console.log('오프라인 준비 완료');
      },
    });
    void pwaUpdate;

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

      await orchestrator.init(savedProvider);
      orchestrator.warmup(['profile_classifier', 'spacing_boundary', 'edit_tagger', 'reranker', 'guardrail']);
    })().catch((e: unknown) => {
      dispatch({ type: 'RUN_ERROR', message: e instanceof Error ? e.message : '초기화 실패' });
    });

    return () => {
      window.removeEventListener('online', onOnline);
      window.removeEventListener('offline', onOffline);
      orchestrator.dispose();
    };
  }, [dispatch, runPipeline]);

  const reinitProvider = useCallback(
    async (provider: 'wasm' | 'webgpu') => {
      if (!orchestratorRef.current) return;
      await orchestratorRef.current.init(provider);
      orchestratorRef.current.warmup(['profile_classifier', 'spacing_boundary', 'edit_tagger', 'reranker', 'guardrail']);
    },
    []
  );

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
          <h1>웹페이지용 온디바이스 한국어 맞춤법 교정 데모</h1>
          <p>파이프라인 계약 기반 시각화 + 오프라인 동작 검증</p>
        </div>
        <div>
          <OfflineBadge
            online={state.offlineSimulation ? false : state.online}
            offlineReady={state.pipelineResult?.assetStatus.offlineCapable ?? false}
          />
          <AssetStatusPanel provider={state.provider} modelReady={modelReady} />
        </div>
      </header>

      {state.error ? <div className="error-banner">오류: {state.error}</div> : null}

      <main className="layout">
        <div className="left-col">
          <EditorPanel
            inputText={state.inputText}
            provider={state.provider}
            offlineSimulation={state.offlineSimulation}
            running={state.running}
            onChangeText={(value, cursor) => dispatch({ type: 'SET_INPUT', value, cursor })}
            onChangeProvider={(value) => {
              dispatch({ type: 'SET_PROVIDER', value });
              void setSetting('provider', value);
              void reinitProvider(value);
            }}
            onToggleOffline={(value) => dispatch({ type: 'SET_OFFLINE_SIM', value })}
            onRun={() => void runPipeline()}
          />

          <FinalOutputPanel
            currentText={state.inputText}
            proposalText={proposalText}
            proposalMarkers={proposalPreview.markers}
            proposalCount={(state.pipelineResult?.edits ?? []).length}
            onApply={() => dispatch({ type: 'SET_INPUT', value: proposalText, cursor: proposalText.length })}
          />
          <CandidatePanel
            edits={state.pipelineResult?.edits ?? []}
            baseText={state.pipelineResult?.original ?? state.inputText}
          />
        </div>

        <div className="right-col">
          <TracePanel traces={state.pipelineResult?.traces ?? []} />
        </div>
      </main>
    </div>
  );
}
