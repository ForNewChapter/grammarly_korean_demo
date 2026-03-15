# Cloud Run Web Setup

목표:
- 프론트는 정적 호스팅
- 교정 API는 Cloud Run
- 브라우저 worker/R2 모델 경로는 사용하지 않음

## 1. FastAPI 이미지 빌드/배포

프로젝트 루트에서:

```bash
gcloud builds submit --tag gcr.io/<GCP_PROJECT_ID>/grammarly-korean-api
```

배포:

```bash
gcloud run deploy grammarly-korean-api \
  --image gcr.io/<GCP_PROJECT_ID>/grammarly-korean-api \
  --platform managed \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 1 \
  --min-instances 0
```

배포 후 서비스 URL 예시:

```text
https://grammarly-korean-api-xxxxxx-uc.a.run.app
```

## 2. 프론트에서 Cloud Run만 사용

Pages/Vercel/GitHub Pages 빌드 시 환경변수:

```bash
VITE_API_BASE_URL=https://grammarly-korean-api-xxxxxx-uc.a.run.app
```

이 값이 있으면 프론트는:
- `/api/health`
- `/api/correct`
를 Cloud Run 쪽으로 호출
- browser worker / R2 모델 경로는 초기화하지 않음

## 3. 동작 확인

브라우저 상단 상태 패널에서:
- `연결: 백엔드 API`

가 보여야 정상입니다.

이 상태에서는:
- Kiwi wasm
- browser ONNX
- R2 remote model
을 사용하지 않습니다.

## 4. 참고

현재 백엔드는 startup에서 모델과 자산을 한 번에 로드하므로 cold start가 있습니다.
권장 시작 사양:
- memory: `8Gi`
- cpu: `2`

이후 실제 사용량을 보고 낮추는 편이 안전합니다.
