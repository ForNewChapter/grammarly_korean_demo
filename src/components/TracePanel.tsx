import { useState } from 'react';
import type { StageTrace, TextEdit } from '../pipeline/types';

interface TracePanelProps {
  traces: StageTrace[];
}

type StageDisplay = {
  title: string;
  description: string;
};

function getStageDisplay(stageName: string): StageDisplay {
  const stages: Array<[RegExp, StageDisplay]> = [
    [/Input Range/i, { title: '1단계. 검사 범위 추출', description: '지금 검사할 문장 범위를 정합니다.' }],
    [/Protected Span/i, { title: '2단계. 보호 구간 탐지', description: 'URL, 숫자, 고유명사처럼 고치면 안 되는 부분을 잠급니다.' }],
    [/Profile Classifier/i, { title: '3단계. 문장 프로파일 분류', description: '문장을 일반 문장인지, 채팅체인지, 오타가 많은지 분류합니다.' }],
    [/Rule Corrector/i, { title: '4단계. 확정 규칙 교정', description: '사전에 있는 확실한 오타를 먼저 바로잡습니다.' }],
    [/Phrase Pre-normalizer/i, { title: '4-1단계. compact 구문 정규화', description: '띄어쓰기 전에 붙어 있는 고정형 표현을 먼저 복원해 뒤 단계가 깨지지 않게 합니다.' }],
    [/Spacing Candidate/i, { title: '5단계. 띄어쓰기 후보 생성', description: '띄어쓰기를 넣을 수 있는 위치를 찾습니다.' }],
    [/Spacing Boundary Classifier/i, { title: '6단계. 띄어쓰기 후보 판정', description: '띄어쓰기 후보 중 실제로 적용할 위치만 남깁니다.' }],
    [/Spacing Edit Converter/i, { title: '6-1단계. 띄어쓰기 수정안 생성', description: '선택된 띄어쓰기 후보를 실제 수정안으로 바꿉니다.' }],
    [/Kiwi Pre-normalizer/i, { title: '6-2단계. 고정밀 정규화', description: 'Kiwi와 로컬 사전으로 자주 틀리는 표면 오류를 먼저 정리합니다.' }],
    [/Kiwi Canonicalizer/i, { title: '6-3단계. canonical state 추출', description: 'Kiwi wasm으로 lemma/POS/활용 슬롯을 추출해 후보 생성 기준 단위를 만듭니다.' }],
    [/Edit Tagger \(/i, { title: '7단계. 오류 감지', description: '문맥상 이상한 토큰이 있는지 보고, 어떤 종류의 교정인지 태깅합니다.' }],
    [/Edit Tagger Routing/i, { title: '7-1단계. 후속 처리 결정', description: '탐지된 오류를 띄어쓰기 체인으로 보낼지, 문맥 교정으로 보낼지 정합니다.' }],
    [/High-precision Lock Guard/i, { title: '7-2단계. 고정밀 수정 보호', description: '앞단에서 확실하게 고친 span은 약한 후속 제안이 다시 뒤집지 못하게 막습니다.' }],
    [/Open Candidate Generation/i, { title: '8단계. 교정 후보 생성', description: '문맥 교정이 필요한 토큰에 대해 실제 교정 후보를 만듭니다.' }],
    [/Candidate Verifier/i, { title: '8-1단계. 후보 1차 정리', description: '생성된 후보 중 믿을 만한 후보만 남깁니다.' }],
    [/Reranker/i, { title: '9단계. 문맥 기준 재정렬', description: '후보들을 문맥 안에서 다시 비교해 가장 자연스러운 후보를 고릅니다.' }],
    [/Guardrail/i, { title: '10단계. 과교정 방지', description: '바꾸면 위험한 수정인지 확인합니다.' }],
    [/Policy Engine/i, { title: '11단계. 적용 정책 결정', description: '자동 반영할지, 제안만 할지, 막을지 결정합니다.' }],
    [/Final Output/i, { title: '12단계. 최종 결과 정리', description: '최종 문장과 수정 건수를 정리합니다.' }],
  ];

  return stages.find(([pattern]) => pattern.test(stageName))?.[1] ?? {
    title: stageName,
    description: '단계 실행 결과입니다.',
  };
}

function profileLabel(profile: string): string {
  const labels: Record<string, string> = {
    NORMAL: '일반 문장',
    CHAT: '채팅체',
    NOISY: '오타가 많은 입력',
    MIXED: '영문/숫자 혼합 입력',
    QUERY: '검색어나 짧은 입력',
  };
  return labels[profile] ?? profile;
}

function tagLabel(label: string): string {
  const labels: Record<string, string> = {
    KEEP: '유지',
    SPACE_FIX: '띄어쓰기 후보',
    OPEN_REPLACE: '문맥 교정 후보',
    PUNCT_FIX: '추가 검토 필요',
    JOSA_FIX: '조사 교정 후보',
    EOMI_FIX: '어미 교정 후보',
  };
  return labels[label] ?? label;
}

function decisionLabel(label: string): string {
  const labels: Record<string, string> = {
    AUTO_APPLY: '자동 반영',
    SUGGEST_ONLY: '제안만',
    REJECT: '차단',
    PROMOTE_TO_OPEN_REPLACE: '문맥 교정 후보로 올림',
    DEFER_TO_SPACING: '띄어쓰기 단계로 넘김',
    DEFER_TO_RULES: '규칙 단계로 넘김',
    PASS_TO_DOWNSTREAM: '후속 단계로 유지',
    LOCKED_HIGH_PRECISION_SPAN: '고정밀 수정 보호로 차단',
  };
  return labels[label] ?? label;
}

function reasonLabel(reason: string): string {
  const labels: Record<string, string> = {
    LEXICAL_REPLACEMENT: '의미가 달라질 수 있어 자동 반영하지 않음',
    PROTECTED_SPAN: '보호 구간이라 수정하지 않음',
    ALNUM_RISK: '영문/숫자 구간이라 수정하지 않음',
    EDIT_RATIO_HIGH: '수정 폭이 커서 보수적으로 처리함',
    CHAT_PROFILE: '채팅체 입력이라 보수적으로 처리함',
  };
  return labels[reason] ?? reason;
}

function insertMarker(text: string, index: number): string {
  return `${text.slice(0, index)}|${text.slice(index)}`;
}

function renderSummary(stageName: string, inputText: string, output: unknown): JSX.Element {
  if (/Input Range/i.test(stageName)) {
    const data = output as { clauseText?: string; clauseRange?: { start: number; end: number }; strategy?: string };
    return (
      <ul className="trace-summary-list">
        <li>검사 범위: <code>{data?.clauseText ?? '-'}</code></li>
        <li>원문 위치: {data?.clauseRange?.start ?? 0} ~ {data?.clauseRange?.end ?? 0}</li>
        <li>검사 방식: {data?.strategy === 'full_text_inspect' ? '입력 전체 검사' : '최근 절만 검사'}</li>
      </ul>
    );
  }

  if (/Protected Span/i.test(stageName)) {
    const data = output as { protected?: Array<{ kind: string; text: string }>; maskedText?: string };
    const items = data?.protected ?? [];
    return items.length ? (
      <div className="trace-summary-block">
        <div className="trace-summary-title">보호된 항목</div>
        <ul className="trace-summary-list">
          {items.map((item, index) => (
            <li key={`${item.kind}-${index}`}>{item.kind}: <code>{item.text}</code></li>
          ))}
        </ul>
      </div>
    ) : (
      <div className="trace-summary-empty">보호 구간이 없습니다.</div>
    );
  }

  if (/Profile Classifier/i.test(stageName)) {
    const data = output as { profile?: string; ruleProfile?: string; modelProfile?: string };
    return (
      <ul className="trace-summary-list">
        <li>최종 프로파일: <strong>{profileLabel(data?.profile ?? 'NORMAL')}</strong></li>
        <li>규칙 판단: {profileLabel(data?.ruleProfile ?? '-')}</li>
        <li>모델 판단: {profileLabel(data?.modelProfile ?? '-')}</li>
      </ul>
    );
  }

  if (/Rule Corrector/i.test(stageName)) {
    const edits = (output as TextEdit[]) ?? [];
    return edits.length ? (
      <ul className="trace-summary-list">
        {edits.map((edit, index) => (
          <li key={`${edit.stage}-${index}`}>{edit.sourceText} → {edit.replacement}</li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">규칙으로 바로 고칠 항목은 없습니다.</div>
    );
  }

  if (/Phrase Pre-normalizer/i.test(stageName)) {
    const edits = (output as TextEdit[]) ?? [];
    return edits.length ? (
      <ul className="trace-summary-list">
        {edits.map((edit, index) => (
          <li key={`phrase-pre-${index}`}>{edit.sourceText} → {edit.replacement}</li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">띄어쓰기 전 미리 고칠 compact 구문은 없습니다.</div>
    );
  }

  if (/Spacing Candidate/i.test(stageName)) {
    const data = output as { spacedText?: string; candidates?: Array<{ index: number }> };
    const candidates = data?.candidates ?? [];
    return (
      <div className="trace-summary-block">
        <div className="trace-summary-title">띄어쓰기 후보 문장</div>
        <div className="trace-summary-inline"><code>{data?.spacedText ?? inputText}</code></div>
        {candidates.length ? (
          <ul className="trace-summary-list">
            {candidates.map((item, index) => (
              <li key={`spacing-candidate-${index}`}>후보 위치: <code>{insertMarker(inputText, item.index)}</code></li>
            ))}
          </ul>
        ) : (
          <div className="trace-summary-empty">띄어쓰기 후보가 없습니다.</div>
        )}
      </div>
    );
  }

  if (/Spacing Boundary Classifier/i.test(stageName)) {
    const items = (output as Array<{ index: number }>) ?? [];
    return items.length ? (
      <ul className="trace-summary-list">
        {items.map((item, index) => (
          <li key={`spacing-accept-${index}`}>채택된 위치: <code>{insertMarker(inputText, item.index)}</code></li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">채택된 띄어쓰기 후보가 없습니다.</div>
    );
  }

  if (/Spacing Edit Converter/i.test(stageName)) {
    const edits = (output as TextEdit[]) ?? [];
    return edits.length ? (
      <ul className="trace-summary-list">
        {edits.map((edit, index) => {
          if (edit.editType === 'SPACE_INSERT') {
            return (
              <li key={`spacing-edit-${index}`}>
                띄어쓰기 추가: <code>{insertMarker(inputText, edit.range.start)}</code> → <code>{`${inputText.slice(0, edit.range.start)} ${inputText.slice(edit.range.start)}`}</code>
              </li>
            );
          }
          return (
            <li key={`spacing-edit-${index}`}>
              띄어쓰기 삭제: <code>{inputText}</code>
            </li>
          );
        })}
      </ul>
    ) : (
      <div className="trace-summary-empty">적용할 띄어쓰기 수정이 없습니다.</div>
    );
  }

  if (/Kiwi Pre-normalizer/i.test(stageName)) {
    const edits = (output as TextEdit[]) ?? [];
    return edits.length ? (
      <ul className="trace-summary-list">
        {edits.map((edit, index) => (
          <li key={`pre-normalize-${index}`}>{edit.sourceText} → {edit.replacement}</li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">고정밀 정규화에서 적용한 수정이 없습니다.</div>
    );
  }

  if (/Kiwi Canonicalizer/i.test(stageName)) {
    const data = output as { ready?: boolean; version?: string; tokenCount?: number; error?: string };
    return (
      <ul className="trace-summary-list">
        <li>준비 상태: {data?.ready ? 'ready' : 'not ready'}</li>
        <li>토큰 수: {data?.tokenCount ?? 0}</li>
        <li>버전: {data?.version ?? '-'}</li>
        <li>오류: {data?.error ?? '-'}</li>
      </ul>
    );
  }

  if (/Edit Tagger \(/i.test(stageName)) {
    const labels = (output as Array<{ token: string; label: string; detectionEvidence?: { bestReplacement?: string } }>) ?? [];
    const issues = labels.filter((item) => item.label !== 'KEEP');
    return issues.length ? (
      <ul className="trace-summary-list">
        {issues.map((item, index) => (
          <li key={`tagger-${index}`}>
            <strong>{item.token}</strong>: {tagLabel(item.label)}
            {item.detectionEvidence?.bestReplacement ? `, 우선 후보 ${item.detectionEvidence.bestReplacement}` : ''}
          </li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">추가로 탐지된 오류가 없습니다.</div>
    );
  }

  if (/Edit Tagger Routing/i.test(stageName)) {
    const labels = (output as Array<{ token: string; effectiveLabel?: string; routingDecision?: string; routingNote?: string }>) ?? [];
    const routed = labels.filter((item) => item.effectiveLabel !== 'KEEP' || item.routingDecision !== 'PASS_TO_DOWNSTREAM');
    return routed.length ? (
      <ul className="trace-summary-list">
        {routed.map((item, index) => (
          <li key={`routing-${index}`}>
            <strong>{item.token}</strong>: {decisionLabel(item.routingDecision ?? 'PASS_TO_DOWNSTREAM')}
            {item.routingNote ? ` - ${item.routingNote}` : ''}
          </li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">후속 처리로 넘어갈 특별한 항목이 없습니다.</div>
    );
  }

  if (/High-precision Lock Guard/i.test(stageName)) {
    const labels = (output as Array<{ token: string; effectiveLabel?: string; routingDecision?: string; routingNote?: string }>) ?? [];
    const guarded = labels.filter((item) => item.routingDecision === 'LOCKED_HIGH_PRECISION_SPAN');
    return guarded.length ? (
      <ul className="trace-summary-list">
        {guarded.map((item, index) => (
          <li key={`lock-guard-${index}`}>
            <strong>{item.token}</strong>: {decisionLabel(item.routingDecision ?? 'LOCKED_HIGH_PRECISION_SPAN')}
            {item.routingNote ? ` - ${item.routingNote}` : ''}
          </li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">앞단 고정밀 수정과 충돌하는 후속 제안은 없었습니다.</div>
    );
  }

  if (/Open Candidate Generation/i.test(stageName)) {
    const groups = (output as Array<{ original: string; items: Array<{ replacement: string; source: string }> }>) ?? [];
    return groups.length ? (
      <div className="trace-summary-block">
        {groups.map((group, index) => (
          <div key={`candidate-group-${index}`} className="trace-group">
            <div className="trace-summary-title"><strong>{group.original}</strong>의 후보</div>
            <ul className="trace-summary-list">
              {group.items
                .filter((item) => item.source !== 'ORIGINAL')
                .map((item, itemIndex) => (
                  <li key={`candidate-${itemIndex}`}>{group.original} → {item.replacement} ({item.source})</li>
                ))}
            </ul>
          </div>
        ))}
      </div>
    ) : (
      <div className="trace-summary-empty">문맥 교정 후보가 없습니다.</div>
    );
  }

  if (/Candidate Verifier/i.test(stageName)) {
    const groups = (output as Array<{ original: string; candidateVerifier?: { inputCount: number; keptCount: number; keptSources: string[] } }>) ?? [];
    return groups.length ? (
      <ul className="trace-summary-list">
        {groups.map((group, index) => (
          <li key={`verifier-${index}`}>
            <strong>{group.original}</strong>: {group.candidateVerifier?.inputCount ?? 0}개 중 {group.candidateVerifier?.keptCount ?? 0}개만 유지
          </li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">검증할 후보가 없습니다.</div>
    );
  }

  if (/Reranker/i.test(stageName)) {
    const groups = (output as Array<{ original: string; best?: { replacement: string; finalScore: number } }>) ?? [];
    return groups.length ? (
      <ul className="trace-summary-list">
        {groups.map((group, index) => (
          <li key={`rerank-${index}`}>
            <strong>{group.original}</strong> → {group.best?.replacement ?? group.original}
            {group.best?.finalScore !== undefined ? ` (점수 ${Math.round(group.best.finalScore * 100)}%)` : ''}
          </li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">재정렬할 후보가 없습니다.</div>
    );
  }

  if (/Guardrail/i.test(stageName)) {
    const items = (output as Array<{ original: string; replacement: string; guardrail?: { decision: string; reasonCodes: string[] } }>) ?? [];
    return items.length ? (
      <ul className="trace-summary-list">
        {items.map((item, index) => (
          <li key={`guardrail-${index}`}>
            <strong>{item.original}</strong> → {item.replacement}: {decisionLabel(item.guardrail?.decision ?? '')}
            {item.guardrail?.reasonCodes?.length ? ` - ${item.guardrail.reasonCodes.map(reasonLabel).join(', ')}` : ''}
          </li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">과교정 검증 대상이 없습니다.</div>
    );
  }

  if (/Policy Engine/i.test(stageName)) {
    const items = (output as Array<{ original: string; replacement: string; decision: string }>) ?? [];
    return items.length ? (
      <ul className="trace-summary-list">
        {items.map((item, index) => (
          <li key={`policy-${index}`}>
            <strong>{item.original}</strong> → {item.replacement}: {decisionLabel(item.decision)}
          </li>
        ))}
      </ul>
    ) : (
      <div className="trace-summary-empty">정책 판단 대상이 없습니다.</div>
    );
  }

  if (/Final Output/i.test(stageName)) {
    const data = output as { finalClause?: string; autoCount?: number; suggestCount?: number; rejectCount?: number };
    return (
      <ul className="trace-summary-list">
        <li>현재 결과 문장: <code>{data?.finalClause ?? '-'}</code></li>
        <li>자동 반영: {data?.autoCount ?? 0}건</li>
        <li>제안만: {data?.suggestCount ?? 0}건</li>
        <li>차단: {data?.rejectCount ?? 0}건</li>
      </ul>
    );
  }

  return <div className="trace-summary-empty">요약할 수 있는 정보가 없습니다.</div>;
}

export function TracePanel({ traces }: TracePanelProps): JSX.Element {
  const [openRaw, setOpenRaw] = useState<Record<string, boolean>>({});

  return (
    <section className="panel trace-panel">
      <h2>단계별 교정 과정</h2>
      <div className="trace-stack">
        {traces.map((trace, index) => {
          const id = `${index}-${trace.stageName}`;
          const showRaw = !!openRaw[id];
          const display = getStageDisplay(trace.stageName);
          return (
            <article className="trace-card" key={id}>
              <header className="trace-head">
                <div>
                  <strong>{display.title}</strong>
                  <p className="why">{display.description}</p>
                </div>
                <span className="trace-latency">{trace.latencyMs}ms</span>
              </header>

              <div className="trace-summary">
                {renderSummary(trace.stageName, trace.inputText, trace.output)}
              </div>

              <button
                className="raw-btn"
                onClick={() => setOpenRaw((prev) => ({ ...prev, [id]: !prev[id] }))}
              >
                {showRaw ? '원본 데이터 닫기' : '원본 데이터 보기'}
              </button>
              {showRaw ? <pre className="raw">{JSON.stringify(trace.output, null, 2)}</pre> : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
