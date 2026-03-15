const DEFAULT_SAMPLE = "오늘날씨가너무좋은데 산책가도되요?";

const inputTextEl = document.querySelector("#inputText");
const idleMsEl = document.querySelector("#idleMs");
const runBtnEl = document.querySelector("#runBtn");
const statusTextEl = document.querySelector("#statusText");
const finalTextEl = document.querySelector("#finalText");
const decisionListEl = document.querySelector("#decisionList");
const decisionPanelEl = document.querySelector("#decisionPanel");
const decisionToggleBtnEl = document.querySelector("#decisionToggleBtn");
const traceListEl = document.querySelector("#traceList");
const modelInfoEl = document.querySelector("#modelInfo");

let idleTimer = null;
let inflightController = null;
let decisionPanelOpen = false;
let decisionCount = 0;

function clampIdleMs(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 900;
  return Math.min(5000, Math.max(100, Math.round(n)));
}

function renderModelInfo(modelInfo) {
  if (!modelInfo) {
    modelInfoEl.textContent = "모델: 정보 없음";
    return;
  }
  modelInfoEl.textContent =
    `모델: spacing=${modelInfo.spacingEngine}, ` +
    `profile=${modelInfo.profileModel}, reranker=${modelInfo.rerankerModel}`;
}

function renderDecisions(decisions) {
  decisionListEl.innerHTML = "";
  const count = decisions?.length || 0;
  decisionCount = count;
  updateDecisionToggle(count);

  if (!decisions || !decisions.length) {
    const li = document.createElement("li");
    li.className = "decision-item";
    li.textContent = "수정 사항이 없습니다.";
    decisionListEl.appendChild(li);
    return;
  }

  for (const d of decisions) {
    const li = document.createElement("li");
    li.className = "decision-item";
    const badgeClass =
      d.decision === "AUTO_APPLY" ? "auto" : d.decision === "SUGGEST_ONLY" ? "suggest" : "reject";
    li.innerHTML =
      `<span class="badge ${badgeClass}">${d.decision}</span>` +
      `[${d.stage}] "${d.sourceText || "∅"}" → "${d.replacement}" ` +
      `(conf ${Number(d.confidence).toFixed(2)} / ${d.reasonTag || "-"})`;
    decisionListEl.appendChild(li);
  }
}

function setDecisionPanelOpen(open) {
  decisionPanelOpen = open;
  decisionPanelEl.classList.toggle("is-collapsed", !open);
  decisionToggleBtnEl.setAttribute("aria-expanded", String(open));
}

function updateDecisionToggle(count) {
  const label = decisionPanelOpen ? "수정 요약 닫기" : "수정 요약 보기";
  decisionToggleBtnEl.textContent = `${label} (${count})`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function stageNameKo(stageName) {
  const table = [
    ["1.", "1단계 입력 범위 추출"],
    ["2.", "2단계 보호 구간 탐지"],
    ["3.", "3단계 문장 프로파일 분류"],
    ["4.", "4단계 규칙 교정"],
    ["5.", "5단계 띄어쓰기 후보 생성"],
    ["6-2.", "6-2단계 띄어쓰기 수정 변환"],
    ["6.", "6단계 띄어쓰기 경계 판정"],
    ["7.", "7단계 오류 태깅"],
    ["8.", "8단계 열린 후보 생성"],
    ["9.", "9단계 후보 랭킹"],
    ["10.", "10단계 가드레일 검증"],
    ["11.", "11단계 정책 결정"],
    ["12.", "12단계 최종 출력"],
  ];
  for (const [prefix, label] of table) {
    if (stageName.startsWith(prefix)) return label;
  }
  return stageName;
}

function shortText(text, limit = 120) {
  if (typeof text !== "string") return "";
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}...`;
}

function chips(values) {
  if (!values.length) return `<span class="trace-muted">없음</span>`;
  return values.map((v) => `<span class="trace-chip">${escapeHtml(v)}</span>`).join(" ");
}

function formatArtifacts(stageName, inputText, output) {
  const inputPreview = shortText(inputText, 140);

  if (stageName.startsWith("1.")) {
    return `
      <div class="trace-row"><strong>입력 미리보기</strong> ${escapeHtml(inputPreview)}</div>
      <div class="trace-row"><strong>검사 범위</strong> ${output?.clauseRange?.start ?? "-"} ~ ${output?.clauseRange?.end ?? "-"}</div>
      <div class="trace-row"><strong>추출된 절</strong> ${escapeHtml(output?.clauseText ?? "")}</div>
    `;
  }
  if (stageName.startsWith("2.")) {
    const protectedList = (output?.protected ?? []).map((p) => `${p.kind}:${p.text}`);
    return `
      <div class="trace-row"><strong>보호 대상</strong> URL, 이메일, 전화번호, 숫자, 해시태그, 멘션, 고유명사, 사용자 사전</div>
      <div class="trace-row"><strong>마스킹 텍스트</strong> ${escapeHtml(output?.maskedText ?? "")}</div>
      <div class="trace-row"><strong>보호 span (${protectedList.length})</strong> ${chips(protectedList)}</div>
    `;
  }
  if (stageName.startsWith("3.")) {
    const lines = [];
    if (output?.profile) lines.push(output.profile);
    return `
      <div class="trace-row"><strong>최종 프로파일</strong> ${chips(lines)}</div>
    `;
  }
  if (stageName.startsWith("4.") || stageName.startsWith("6-2.")) {
    const edits = (output ?? []).map((e) => `${e.sourceText || "∅"} → ${e.replacement}`);
    return `
      <div class="trace-row"><strong>수정 개수</strong> ${edits.length}개</div>
      <div class="trace-row"><strong>수정 내용</strong> ${chips(edits)}</div>
    `;
  }
  if (stageName.startsWith("5.")) {
    const candidates = (output?.candidates ?? []).map((c) => `${c.action}@${c.index} (${c.score})`);
    return `
      <div class="trace-row"><strong>사용 엔진</strong> Kiwi (<code>kiwipiepy</code>) - 보호 span 제외 구간의 띄어쓰기 후보 생성</div>
      <div class="trace-row"><strong>띄어쓰기 결과(참고)</strong> ${escapeHtml(output?.spacedText ?? "")}</div>
      <div class="trace-row"><strong>후보 (${candidates.length})</strong> ${chips(candidates)}</div>
    `;
  }
  if (stageName.startsWith("6.")) {
    const boundaries = (output ?? []).map((b) => `${b.action}@${b.index} (${b.score})`);
    return `
      <div class="trace-row"><strong>채택 경계 (${boundaries.length})</strong> ${chips(boundaries)}</div>
    `;
  }
  if (stageName.startsWith("7.")) {
    const labels = (output ?? []).map((o) => `${o.token}:${o.label}`);
    return `
      <div class="trace-row"><strong>태깅 방식</strong> 룰 기반 태깅 + KoBERT-MLM 점수 보조(토큰 후보 비교)</div>
      <div class="trace-row"><strong>태그 결과 (${labels.length})</strong> ${chips(labels)}</div>
    `;
  }
  if (stageName.startsWith("8.")) {
    const groups = (output ?? []).map((g) => {
      const cands = (g.items || []).map((i) => i.replacement).join(", ");
      return `${g.original} -> [${cands}]`;
    });
    return `
      <div class="trace-row"><strong>후보 그룹 (${groups.length})</strong> ${chips(groups)}</div>
    `;
  }
  if (stageName.startsWith("9.")) {
    const ranked = (output ?? []).map((g) => `${g.original} -> ${g.best?.replacement} (${g.best?.finalScore})`);
    return `
      <div class="trace-row"><strong>랭킹 결과</strong> ${chips(ranked)}</div>
    `;
  }
  if (stageName.startsWith("10.")) {
    const guards = (output ?? []).map((g) => `${g.original} -> ${g.guardrail?.decision}`);
    return `
      <div class="trace-row"><strong>가드레일</strong> ${chips(guards)}</div>
    `;
  }
  if (stageName.startsWith("11.")) {
    const policy = (output ?? []).map((g) => `${g.original} -> ${g.decision}`);
    return `
      <div class="trace-row"><strong>정책 결정</strong> ${chips(policy)}</div>
    `;
  }
  if (stageName.startsWith("12.")) {
    return `
      <div class="trace-row"><strong>최종 절</strong> ${escapeHtml(output?.finalClause ?? "")}</div>
      <div class="trace-row"><strong>요약</strong> 자동 ${output?.autoCount ?? 0} / 제안 ${
      output?.suggestCount ?? 0
    } / 차단 ${output?.rejectCount ?? 0}</div>
    `;
  }
  return `<div class="trace-row">${escapeHtml(JSON.stringify(output))}</div>`;
}

function renderTraces(traces) {
  traceListEl.innerHTML = "";
  for (const t of traces || []) {
    const card = document.createElement("article");
    card.className = "trace-card";
    const head = document.createElement("div");
    head.className = "trace-head";
    head.textContent = `${stageNameKo(t.stageName)} (${t.latencyMs} ms)`;
    const body = document.createElement("div");
    body.className = "trace-body";
    body.innerHTML = formatArtifacts(t.stageName, t.inputText, t.outputArtifacts);
    card.append(head, body);
    traceListEl.appendChild(card);
  }
}

async function ensureHealth() {
  const resp = await fetch("/api/health", { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`health check failed (${resp.status})`);
  }
  const body = await resp.json();
  if (!body.ok) {
    throw new Error("model engine is not ready");
  }
}

async function runCorrection() {
  const text = inputTextEl.value;
  let cursor = inputTextEl.selectionStart ?? text.length;
  if (document.activeElement !== inputTextEl && cursor === 0 && text.length > 0) {
    cursor = text.length;
  }
  const idleMs = clampIdleMs(idleMsEl.value);
  idleMsEl.value = String(idleMs);

  if (inflightController) inflightController.abort();
  inflightController = new AbortController();

  statusTextEl.textContent = "교정 실행 중... (로컬 API)";
  try {
    const resp = await fetch("/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        cursor,
        mode: "realtime",
      }),
      signal: inflightController.signal,
    });
    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`${resp.status} ${body}`);
    }
    const result = await resp.json();
    finalTextEl.textContent = result.finalText || "";
    renderModelInfo(result.modelInfo);
    renderDecisions(result.decisions);
    renderTraces(result.traces);
    statusTextEl.textContent = `완료 (${new Date().toLocaleTimeString()})`;
  } catch (err) {
    if (err.name === "AbortError") return;
    statusTextEl.textContent = "오류: 로컬 API 연결 실패";
    finalTextEl.textContent = "";
    decisionListEl.innerHTML = "";
    decisionCount = 0;
    updateDecisionToggle(0);
    setDecisionPanelOpen(false);
    traceListEl.innerHTML = "";
    modelInfoEl.textContent = "모델: 로컬 API 미연결";
    const li = document.createElement("li");
    li.className = "decision-item";
    li.textContent = `오류: ${err.message}`;
    decisionListEl.appendChild(li);
  }
}

function scheduleRun() {
  const idleMs = clampIdleMs(idleMsEl.value);
  idleMsEl.value = String(idleMs);
  if (idleTimer) clearTimeout(idleTimer);
  statusTextEl.textContent = `${idleMs}ms 대기 중...`;
  idleTimer = setTimeout(() => {
    runCorrection();
  }, idleMs);
}

async function bootstrap() {
  if (!inputTextEl.value.trim()) {
    inputTextEl.value = DEFAULT_SAMPLE;
  }
  modelInfoEl.textContent = "모델: 로컬 API 모드 (Kiwi + KoBERT)";
  try {
    await ensureHealth();
    runCorrection();
  } catch (err) {
    statusTextEl.textContent = "오류: 로컬 API를 먼저 실행하세요";
    const li = document.createElement("li");
    li.className = "decision-item";
    li.textContent =
      "서버 실행 필요: source .venv/bin/activate && uvicorn server:app --host 127.0.0.1 --port 5173";
    decisionListEl.innerHTML = "";
    decisionListEl.appendChild(li);
    decisionCount = 0;
    updateDecisionToggle(0);
    setDecisionPanelOpen(false);
    modelInfoEl.textContent = "모델: 로컬 API 미연결";
  }
}

inputTextEl.addEventListener("input", scheduleRun);
runBtnEl.addEventListener("click", runCorrection);
idleMsEl.addEventListener("change", scheduleRun);
decisionToggleBtnEl.addEventListener("click", () => {
  setDecisionPanelOpen(!decisionPanelOpen);
  updateDecisionToggle(decisionCount);
});

setDecisionPanelOpen(false);
updateDecisionToggle(0);

bootstrap();
