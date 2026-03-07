interface AssetStatusPanelProps {
  provider: 'wasm' | 'webgpu';
  modelReady: Record<string, boolean>;
}

export function AssetStatusPanel({ provider, modelReady }: AssetStatusPanelProps): JSX.Element {
  const entries = Object.entries(modelReady);
  const readyCount = entries.filter(([, ok]) => ok).length;

  return (
    <div className="asset-status">
      <div><strong>Provider:</strong> {provider}</div>
      <div><strong>Models:</strong> {readyCount}/{entries.length || 0}</div>
      <div className="chip-wrap">
        {entries.map(([k, ok]) => (
          <span className={`chip ${ok ? 'ok' : 'muted'}`} key={k}>{k}:{ok ? 'ready' : 'missing'}</span>
        ))}
      </div>
    </div>
  );
}
