interface OfflineBadgeProps {
  online: boolean;
  offlineReady: boolean;
}

export function OfflineBadge({ online, offlineReady }: OfflineBadgeProps): JSX.Element {
  return (
    <div className="badge-row">
      <span className={`pill ${online ? 'ok' : 'warn'}`}>{online ? '온라인' : '오프라인'}</span>
      <span className={`pill ${offlineReady ? 'ok' : 'muted'}`}>{offlineReady ? '오프라인 가능' : '오프라인 준비중'}</span>
    </div>
  );
}
