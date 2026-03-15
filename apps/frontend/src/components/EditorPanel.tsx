interface EditorPanelProps {
  inputText: string;
  running: boolean;
  runtimeMode?: 'browser' | 'backend';
  onChangeText: (value: string, cursor: number) => void;
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
            <span className="readonly-chip">{props.runtimeMode === 'backend' ? '백엔드 API' : '브라우저 로컬'}</span>
          </label>
        </div>
      </div>

      <textarea
        className="editor"
        value={props.inputText}
        placeholder="문장을 입력한 뒤 '맞춤법 검사'를 누르세요."
        onChange={(e) => props.onChangeText(e.target.value, e.target.selectionStart)}
        onClick={(e) => props.onChangeText((e.target as HTMLTextAreaElement).value, (e.target as HTMLTextAreaElement).selectionStart)}
      />

      <div className="controls-inline">
        <span />
        <button onClick={props.onRun} disabled={props.running}>{props.running ? '검사 중...' : '맞춤법 검사'}</button>
      </div>
    </section>
  );
}
