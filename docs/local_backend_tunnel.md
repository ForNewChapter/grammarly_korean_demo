# Hosted Frontend + Local Backend Tunnel

이 프로젝트의 현재 권장 팀 테스트 구조는 아래입니다.

- 프론트: GitHub Pages
- 백엔드: 내 로컬 컴퓨터에서 FastAPI 실행
- 외부 노출: Cloudflare Quick Tunnel

즉, 웹사이트는 정적으로 배포하고 교정 계산은 내 컴퓨터가 수행합니다.

## 왜 이 구조를 쓰는가

현재 백엔드는 startup 시 `Kiwi + KoBERT + KoBERT-MLM + KoELECTRA + local edit tagger`를 로드합니다.
이 구조는 Cloud Run 같은 scale-to-zero 환경보다, 로컬에서 계속 켜두는 방식이 더 단순합니다.

## 사전 조건

### 1. Python 환경

프로젝트 루트에서:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r apps/backend/requirements.txt
```

### 2. cloudflared 설치

macOS:

```bash
brew install cloudflared
```

설치 확인:

```bash
cloudflared --version
```

## 실행 절차

### 1. 백엔드 실행

프로젝트 루트에서:

```bash
npm run backend:dev
```

기본 주소:

```text
http://127.0.0.1:8000
```

이 터미널은 계속 켜둡니다.

### 2. 헬스체크 확인

새 터미널:

```bash
curl http://127.0.0.1:8000/api/health
```

정상 예시:

```json
{"ok": true, "engineReady": false, "engineInitializing": true, "engineError": null, "cloudRun": false}
```

`ok`는 서버가 떴다는 뜻입니다.  
`engineReady`는 무거운 교정 엔진이 준비됐는지 뜻합니다.

### 3. Quick Tunnel 실행

새 터미널:

```bash
npm run tunnel:quick
```

정상 예시:

```text
https://abc-def-ghi.trycloudflare.com
```

이 주소가 외부에서 접근 가능한 백엔드 URL입니다.

### 4. 팀원에게 보낼 최종 링크 만들기

GitHub Pages 주소가:

```text
https://fornewchapter.github.io/grammarly_korean_demo/
```

Quick Tunnel 주소가:

```text
https://abc-def-ghi.trycloudflare.com
```

이면:

```bash
npm run share:url -- https://fornewchapter.github.io/grammarly_korean_demo/ https://abc-def-ghi.trycloudflare.com
```

출력 예시:

```text
https://fornewchapter.github.io/grammarly_korean_demo/?apiBase=https%3A%2F%2Fabc-def-ghi.trycloudflare.com
```

이 링크를 팀원에게 공유하면 됩니다.

## 브라우저에서 확인할 것

상단 상태 패널에 아래가 보여야 합니다.

- `연결: 백엔드 API`
- `백엔드: <이름> / <브랜치> / <버전>`
- `주소: https://...trycloudflare.com`

백엔드 엔진이 아직 초기화 중이면 `초기화 중`, 실패했으면 `오류`로 표시됩니다.

## 운영 주의점

### 1. 네 컴퓨터가 켜져 있어야 함

- 맥북 절전 모드 들어가면 끊깁니다.
- 백엔드 터미널과 tunnel 터미널 둘 다 살아 있어야 합니다.

### 2. Quick Tunnel 주소는 임시

`cloudflared`를 다시 켜면 주소가 바뀔 수 있습니다.
그때는 새 링크를 다시 공유해야 합니다.

### 3. 테스트용

이 구조는 팀 테스트/데모에는 적합하지만, 상용 운영용 구조는 아닙니다.

## 팀원별 백엔드 교체

이 구조에서는 프론트가 고정이고, 실제 교정 엔진은 **누가 어떤 백엔드를 띄우느냐**에 따라 달라집니다.

- 팀원 A가 자기 브랜치 백엔드를 띄우면 A 버전 링크
- 팀원 B가 자기 브랜치 백엔드를 띄우면 B 버전 링크

즉 팀원도 각자:

1. 자기 브랜치 checkout
2. `npm run backend:dev`
3. `npm run tunnel:quick`
4. `npm run share:url -- ...`

순서로 자기 버전을 공유할 수 있습니다.

프론트가 안정적으로 붙으려면 아래 API 계약은 유지하는 것이 좋습니다.

- [docs/backend_api_contract.md](/Users/joh/Desktop/joh9911/MyProject/grammarly_korean/docs/backend_api_contract.md)

## 문제 해결

### `api/health`가 안 됨

백엔드 터미널에서 에러를 먼저 확인합니다.

```bash
curl http://127.0.0.1:8000/api/health
```

이게 안 되면 tunnel 문제가 아니라 로컬 백엔드 문제입니다.

### 터널 주소는 떴는데 웹에서 연결 안 됨

브라우저에서 이 주소를 직접 열어봅니다.

```text
https://abc-def-ghi.trycloudflare.com/api/health
```

이게 안 되면 tunnel/CORS 문제입니다.

### GitHub Pages는 뜨는데 교정이 안 됨

공유 링크에 `?apiBase=...`가 붙어 있는지 확인합니다.

프론트는 `apiBase`가 있으면 브라우저 모델/R2 경로를 쓰지 않고, 백엔드 API만 사용합니다.
