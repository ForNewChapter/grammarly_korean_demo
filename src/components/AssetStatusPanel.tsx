interface AssetStatusPanelProps {
  provider: 'wasm' | 'webgpu';
  modelReady: Record<string, boolean>;
  rulesReady?: boolean;
  kiwiReady?: boolean;
  assetErrors?: string[];
}

export function AssetStatusPanel({
  provider,
  modelReady,
  rulesReady = false,
  kiwiReady = false,
  assetErrors = [],
}: AssetStatusPanelProps): JSX.Element {
  const entries = Object.entries(modelReady);
  const readyCount = entries.filter(([, ok]) => ok).length;

  return (
    <div className="asset-status">
      <div><strong>Provider:</strong> {provider}</div>
      <div><strong>Models:</strong> {readyCount}/{entries.length || 0}</div>
      <div><strong>Rules:</strong> {rulesReady ? 'ready' : 'loading/missing'}</div>
      <div><strong>Kiwi:</strong> {kiwiReady ? 'ready' : 'loading/missing'}</div>
      <div className="chip-wrap">
        {entries.map(([k, ok]) => (
          <span className={`chip ${ok ? 'ok' : 'muted'}`} key={k}>{k}:{ok ? 'ready' : 'missing'}</span>
        ))}
      </div>
      {assetErrors.length ? <div className="error-banner">assets: {assetErrors[0]}</div> : null}
    </div>
  );
}
