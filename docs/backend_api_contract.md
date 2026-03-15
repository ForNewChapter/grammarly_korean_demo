# Backend API Contract

이 문서는 **hosted frontend + swap 가능한 local backend** 구조에서 프론트가 기대하는 최소 API 계약을 정의합니다.

목적:

- 팀원이 각자 백엔드 방법론을 수정하더라도
- 프론트는 재배포 없이 같은 방식으로 붙을 수 있게 하기

## 원칙

백엔드는 내부 교정 로직을 자유롭게 바꿀 수 있습니다.
하지만 아래 API 계약은 가능한 한 유지해야 합니다.

## 1. `GET /api/health`

프론트가 백엔드 연결 상태를 확인할 때 사용합니다.

### 최소 응답 형식

```json
{
  "ok": true,
  "backendName": "johs-macbook-pro",
  "backendBranch": "feature/reranker-v2",
  "backendVersion": "a1b2c3d-dirty",
  "engineReady": true,
  "engineInitializing": false,
  "engineError": null,
  "cloudRun": false
}
```

### 필드 의미

- `ok`
  - HTTP 서비스 자체가 살아 있는지
- `backendName`
  - 현재 붙은 백엔드 식별 이름
- `backendBranch`
  - 현재 백엔드가 실행 중인 브랜치
- `backendVersion`
  - 현재 백엔드 버전 또는 커밋 식별자
- `engineReady`
  - 교정 엔진이 실제로 준비됐는지
- `engineInitializing`
  - 무거운 모델/자산 초기화가 진행 중인지
- `engineError`
  - 초기화 실패 시 에러 문자열
- `cloudRun`
  - 선택적 운영 메타데이터

### 프론트가 실제로 쓰는 필드

- `ok`
- `backendName`
- `backendBranch`
- `backendVersion`
- `engineReady`
- `engineInitializing`
- `engineError`

즉 이 필드들은 유지하는 것이 좋습니다.

## 2. `POST /api/correct`

프론트가 실제 교정 요청을 보낼 때 사용합니다.

### 요청 형식

```json
{
  "text": "문장을 입력합니다.",
  "cursor": 10,
  "mode": "inspect"
}
```

### 요청 필드

- `text: string`
- `cursor: number | null`
- `mode: "realtime" | "inspect"`

### 최소 응답 형식

```json
{
  "finalText": "자동 반영 결과 문장",
  "suggestedText": "제안 포함 결과 문장",
  "decisions": [],
  "traces": []
}
```

### 프론트가 직접 소비하는 주요 필드

- `finalText`
- `suggestedText`
- `decisions`
- `traces`

### `decisions` 최소 기대 구조

프론트는 각 수정 건을 아래 형태로 기대합니다.

```json
{
  "stage": "RERANKER",
  "range": { "start": 0, "end": 2 },
  "sourceText": "되요",
  "replacement": "돼요",
  "editType": "SPELL",
  "confidence": 0.98,
  "reasonTag": "common_misspelling",
  "decision": "AUTO_APPLY"
}
```

### `traces` 최소 기대 구조

```json
{
  "stageName": "Rule Corrector",
  "inputText": "원문",
  "outputArtifacts": {},
  "latencyMs": 1.5
}
```

프론트는:

- `stageName`
- `inputText`
- `outputArtifacts` 또는 `output`
- `latencyMs`

를 읽어 단계 UI를 구성합니다.

## 무엇을 자유롭게 바꿔도 되나

아래는 자유롭게 바꿔도 됩니다.

- candidate generation 방식
- verifier/reranker 방식
- Kiwi 활용 방식
- KoELECTRA/KoBERT/MLM 사용 여부
- rules / graph / phrase memory 자산
- 내부 trace 상세 필드

즉 **방법론 자체는 자유롭게 바뀌어도 됩니다.**

## 무엇을 함부로 깨면 안 되나

아래가 바뀌면 프론트도 같이 수정해야 합니다.

- `/api/health` 존재 여부
- `/api/correct` 존재 여부
- `POST /api/correct` 요청 body 형식
- `finalText`, `suggestedText`, `decisions`, `traces`의 기본 의미

## 팀 운영 권장 규칙

### 백엔드만 수정하는 실험

- 프론트 재배포 불필요
- 자기 브랜치에서 백엔드 실행
- Quick Tunnel 링크 공유

### 프론트 UI까지 수정하는 실험

- 프론트 재배포 또는 로컬 프론트 실행 필요

### API 계약을 바꾸는 실험

- 프론트와 백엔드를 함께 수정해야 함
- 가급적 `develop`에서 통합 검증 후 공유
