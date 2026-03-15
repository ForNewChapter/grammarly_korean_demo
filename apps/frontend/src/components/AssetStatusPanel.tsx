interface AssetStatusPanelProps {
  provider: 'wasm' | 'webgpu';
  modelReady: Record<string, boolean>;
  runtimeMode?: 'browser' | 'backend';
  modelSource?: Record<string, 'local' | 'bundled' | 'remote' | 'backend' | 'missing'>;
  rulesReady?: boolean;
  kiwiReady?: boolean;
  assetErrors?: string[];
  backend?: {
    apiBaseUrl: string;
    backendName?: string | null;
    backendBranch?: string | null;
    backendVersion?: string | null;
    engineReady?: boolean;
    engineInitializing?: boolean;
    engineError?: string | null;
  };
}

export function AssetStatusPanel({
  provider,
  modelReady,
  runtimeMode = 'browser',
  modelSource = {},
  rulesReady = false,
  kiwiReady = false,
  assetErrors = [],
  backend,
}: AssetStatusPanelProps): JSX.Element {
  const entries = Object.entries(modelReady);
  const readyCount = entries.filter(([, ok]) => ok).length;
  const usingLocal = entries.some(([key, ok]) => ok && modelSource[key] === 'local');
  const usingBundled = entries.some(([key, ok]) => ok && modelSource[key] === 'bundled');
  const usingRemote = entries.some(([key, ok]) => ok && modelSource[key] === 'remote');
  const backendEngineState = backend?.engineReady
    ? '연결됨'
    : backend?.engineInitializing
      ? '초기화 중'
      : backend?.engineError
        ? '오류'
        : '대기 중';
  const backendIdentity = [backend?.backendName, backend?.backendBranch, backend?.backendVersion]
    .filter((value): value is string => !!value)
    .join(' / ');

  return (
    <div className="asset-status">
      <div>
        <strong>연결:</strong>{' '}
        {runtimeMode === 'backend'
          ? `백엔드 API (${backendEngineState})`
          : usingLocal
            ? 'local models'
            : usingBundled
              ? 'bundled models'
              : usingRemote
                ? 'remote models'
                : 'heuristic fallback'}
      </div>
      {runtimeMode === 'backend' && backend ? (
        <>
          <div className="hint-text"><strong>백엔드:</strong> {backendIdentity || '식별 정보 없음'}</div>
          <div className="hint-text"><strong>주소:</strong> {backend.apiBaseUrl}</div>
          {backend.engineError ? <div className="hint-text">오류: {backend.engineError}</div> : null}
        </>
      ) : (
        <>
          <div className="hint-text"><strong>Provider:</strong> {provider}</div>
          <div className="hint-text"><strong>Models:</strong> {readyCount}/{entries.length || 0}</div>
          <div className="hint-text"><strong>Rules / Kiwi:</strong> {rulesReady ? 'ready' : 'loading'} / {kiwiReady ? 'ready' : 'loading'}</div>
        </>
      )}
      {assetErrors.length ? <div className="error-banner">assets: {assetErrors[0]}</div> : null}
    </div>
  );
}
