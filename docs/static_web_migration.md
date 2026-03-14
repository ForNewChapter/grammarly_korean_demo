# Static Web Migration

## Goal

Run the grammar correction demo as a static web app with:

- no required `/api/correct` backend
- browser worker as the main pipeline
- ONNX models + cached JSON assets in the client
- zero central inference server cost

## Current status

Implemented in this tranche:

1. Browser worker is the default runtime path.
2. Local API is opt-in only via `VITE_PREFER_LOCAL_API=1` or `?localApi=1`.
3. Worker loads runtime assets directly from `/assets/...`.
4. Worker uses asset-driven:
   - surface fixes
   - phrase pre-normalization
   - candidate generation
5. Kiwi wasm runs inside the worker for canonical lemma/POS extraction and reinflection.
6. UI exposes runtime asset readiness/errors, including Kiwi readiness.

## Still pending for full parity

1. richer canonical-state routing beyond current lemma lookup
2. stronger reinflection coverage and slot-aware verifier parity
3. closer reranker feature parity with the Python backend
4. lighter-weight Kiwi model choice to reduce first-run payload

## Static run

```bash
npm install
npm run build
npm run preview
```

Notes:

- `npm run build` and `npm run dev` now call `npm run prepare:kiwi` first.
- `prepare:kiwi` downloads the official Kiwi wasm model files into `public/assets/kiwi/model/`.
- The first browser session still downloads a large Kiwi model set (roughly 100MB+), so initial `inspect` is heavier than later cached runs.

## Local API comparison mode

```bash
VITE_PREFER_LOCAL_API=1 npm run dev
```

or append:

```text
?localApi=1
```

## Deployment targets

Static hosting only:

- GitHub Pages
- Cloudflare Pages
- Vercel Hobby
