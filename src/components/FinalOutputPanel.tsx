import { renderHighlightedText } from '../utils/editPreview';
import type { HighlightMarker } from '../utils/editPreview';

interface FinalOutputPanelProps {
  currentText: string;
  proposalText: string;
  proposalMarkers: HighlightMarker[];
  proposalCount: number;
  onApply: () => void;
}

export function FinalOutputPanel({
  currentText,
  proposalText,
  proposalMarkers,
  proposalCount,
  onApply,
}: FinalOutputPanelProps): JSX.Element {
  const hasProposal = proposalText !== currentText;

  return (
    <section className="panel">
      <div className="proposal-head">
        <div>
          <h2>교정 제안</h2>
          <p className="proposal-copy">
            검사 결과를 하나의 제안 문장으로 합쳐 보여줍니다.
          </p>
        </div>
        <button
          className="primary-apply-btn"
          type="button"
          onClick={onApply}
          disabled={!hasProposal}
        >
          제안 적용
        </button>
      </div>

      <div className="result-grid">
        <div>
          <h3>현재 입력</h3>
          <pre className="box">{currentText || '-'}</pre>
        </div>
        <div>
          <h3>제안 문장</h3>
          <div className="box suggestion-box">
            {renderHighlightedText(proposalText || '-', proposalMarkers)}
          </div>
        </div>
      </div>

      <div className="proposal-footnote">
        <span className="pill warn">포함된 수정 {proposalCount}개</span>
        <span>버튼을 누르면 입력창 문장이 이 제안 문장으로 교체됩니다.</span>
      </div>
    </section>
  );
}
