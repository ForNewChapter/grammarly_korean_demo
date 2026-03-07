interface EditorPanelProps {
  inputText: string;
  provider: 'wasm' | 'webgpu';
  offlineSimulation: boolean;
  running: boolean;
  onChangeText: (value: string, cursor: number) => void;
  onChangeProvider: (v: 'wasm' | 'webgpu') => void;
  onToggleOffline: (v: boolean) => void;
  onRun: () => void;
}

export function EditorPanel(props: EditorPanelProps): JSX.Element {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>입력</h2>
        <div className="controls-inline">
          <label>
            실행 방식
            <span className="readonly-chip">버튼으로 검사</span>
          </label>
          <label>
            Provider
            <select value={props.provider} onChange={(e) => props.onChangeProvider(e.target.value as 'wasm' | 'webgpu')}>
              <option value="wasm">wasm</option>
              <option value="webgpu">webgpu</option>
            </select>
          </label>
        </div>
      </div>

      <textarea
        className="editor"
        value={props.inputText}
        placeholder="문장을 입력한 뒤 '지금 검사'를 누르세요."
        onChange={(e) => props.onChangeText(e.target.value, e.target.selectionStart)}
        onClick={(e) => props.onChangeText((e.target as HTMLTextAreaElement).value, (e.target as HTMLTextAreaElement).selectionStart)}
      />

      <div className="controls-inline">
        <label className="toggle">
          <input
            type="checkbox"
            checked={props.offlineSimulation}
            onChange={(e) => props.onToggleOffline(e.target.checked)}
          />
          오프라인 시뮬레이션
        </label>
        <button onClick={props.onRun} disabled={props.running}>{props.running ? '검사 중...' : '지금 검사'}</button>
      </div>
    </section>
  );
}
