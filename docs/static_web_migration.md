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

1. real browser exports for profile / reranker / guardrail paths
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

- `npm run build` and `npm run dev` now call `npm run prepare:runtime` first.
- `prepare:runtime` runs both:
  - `prepare:kiwi` → downloads Kiwi wasm assets into `public/assets/kiwi/model/`
  - `prepare:models` → copies real local ONNX files from `.local-assets/models/` into `public/assets/models-local/`
- browser edit tagger has a separate export path:
  - `npm run prepare:browser-edit-tagger:local`
  - `npm run prepare:browser-edit-tagger:bundled`
- exported browser edit tagger assets can be smoke-tested with:
  - `npm run smoke:browser-edit-tagger`
- The first browser session still downloads a large Kiwi model set (roughly 100MB+), so initial `inspect` is heavier than later cached runs.
- A normal refresh does not discard downloaded assets. Kiwi/model files are cached via the service worker and reused on later visits until browser data is cleared or a newer build invalidates them.

## Teammate local branch workflow

Use this when teammates are testing their own branches without a central inference server.

1. Put real ONNX files in:

```text
.local-assets/models/
```

Expected files:

- `profile_classifier.onnx`
- `spacing_boundary.onnx`
- `edit_tagger.onnx`
- `reranker.onnx`
- `guardrail.onnx`

2. Prepare and validate:

```bash
npm install
npm run prepare:runtime
npm run prepare:browser-edit-tagger:local
npm run smoke:browser-edit-tagger
npm run validate:runtime:strict
```

3. Run locally:

```bash
npm run dev
```

Runtime selection order:

1. `/assets/models-local/*.onnx`
2. `/assets/models/*.onnx`
3. heuristic fallback if both are missing or placeholder-sized

Important:

- A remotely hosted static site cannot read each teammate's local disk.
- This workflow is for **local branch execution**, not remote Pages deployment.

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
