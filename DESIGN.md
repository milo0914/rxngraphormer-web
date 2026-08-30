# RXNGraphormer Web — Design Specification

**Version:** 1.0  
**Status:** Draft (US-009)  
**Scope:** Backend (US-010), Frontend (US-011), Deployment (US-012)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  GitHub Pages → Static Frontend (HTML/JS/CSS, no build)     │   │
│  │  • Two tabs: Forward / Retrosynthesis                       │   │
│  │  • SMILES textarea + Ketcher drawer (iframe)                │   │
│  │  • Predict → calls Backend API                               │   │
│  │  • Results: ranked cards (score, SMILES, molecule image)    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                    HTTPS REST (configurable base URL)              │
│                              ▼                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (VPS → GCP Cloud Run)           │
│  • Endpoints: /health, /predict/forward, /predict/retro           │
│  • Loads checkpoints on startup (GitHub Releases → figshare)      │
│  • Runs inference on CPU (torch 2.1.2+cpu)                        │
│  • Returns JSON: ranked candidates + scores + base64 PNG images   │
│  • Config via env vars (model paths, device, beam, API URL)       │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principles:**
- **No build step** for frontend — single `index.html` + `config.js` served by GitHub Pages
- **Configurable API base URL** — frontend reads `window.BACKEND_URL` from `config.js`; backend reads `BACKEND_URL` for CORS
- **Checkpoints fetched at startup** — `download_models.py` downloads from GitHub Releases (fallback figshare) into `models/`
- **Simplicity first** — minimal deps, CPU-only, single Dockerfile for backend

---

## 2. Backend Specification (FastAPI)

### 2.1 Directory Structure (Backend)

```
backend/
├── main.py              # FastAPI app, routes, startup/shutdown
├── config.py            # Pydantic Settings (env vars)
├── models/
│   ├── __init__.py
│   ├── loader.py        # Checkpoint loading (reuses US-006/007 logic)
│   ├── forward.py       # Forward inference wrapper
│   └── retro.py         # Retro inference wrapper
├── schemas.py           # Pydantic request/response models
├── rendering.py         # RDKit molecule → base64 PNG
├── download_models.py   # Startup model fetcher (see §6)
├── requirements.txt     # Backend deps
├── Dockerfile           # Container for VPS/GCP
└── run.sh               # Local dev runner
```

### 2.2 Environment Variables (`.env` or Docker `-e`)

| Variable | Default | Description |
|---|---|---|
| `BACKEND_HOST` | `0.0.0.0` | Bind host |
| `BACKEND_PORT` | `8000` | Bind port |
| `BACKEND_URL` | `http://localhost:8000` | Public base URL (for CORS + frontend config) |
| `FORWARD_MODEL_DIR` | `models/seq-v2-USPTO_STEREO-20250509_070206_ft` | Forward checkpoint dir |
| `RETRO_MODEL_DIR` | `models/USPTO_50k` | Retro checkpoint dir |
| `FORWARD_VOCAB_PATH` | `dataset/USPTO_STEREO/vocab_smiles.txt` | Forward decoder vocab |
| `RETRO_VOCAB_PATH` | `dataset/USPTO_50k/vocab_smiles.txt` | Retro decoder vocab |
| `DEVICE` | `cpu` | `cpu` or `cuda:0` |
| `BEAM_SIZE` | `20` | Beam search width |
| `N_BEST` | `10` | Max candidates returned |
| `TOP_K_DEFAULT` | `5` | Default `top_k` if not provided |
| `DOWNLOAD_MODELS_ON_START` | `true` | Run `download_models.py` at startup |
| `GITHUB_RELEASES_REPO` | `milo0914/rxngraphormer-web` | Repo for Release assets |
| `GITHUB_TOKEN` | *(optional)* | PAT for private releases / rate limits |

### 2.3 API Contract

#### Health Check
```
GET /health
```
**Response (200):**
```json
{
  "status": "ok",
  "forward_model_loaded": true,
  "retro_model_loaded": true,
  "device": "cpu"
}
```

#### Forward Prediction
```
POST /predict/forward
Content-Type: application/json
```
**Request:**
```json
{
  "reactants": "CCO.CC(=O)O",
  "top_k": 5,
  "beam_size": 20,
  "return_images": true
}
```
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `reactants` | string | ✅ | — | Reactant SMILES (dot-separated for multiple) |
| `top_k` | int | ❌ | `TOP_K_DEFAULT` (5) | Number of ranked predictions to return (≤ `N_BEST`) |
| `beam_size` | int | ❌ | `BEAM_SIZE` (20) | Beam width for search |
| `return_images` | bool | ❌ | `true` | Include base64 PNG in response |

**Success Response (200):**
```json
{
  "predictions": [
    {
      "rank": 1,
      "smiles": "CCOC(=O)CC",
      "score": -0.123456,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgA...",
      "image_mime": "image/png"
    },
    ...
  ],
  "request_id": "fwd_abc123"
}
```

#### Retrosynthesis Prediction
```
POST /predict/retro
Content-Type: application/json
```
**Request:**
```json
{
  "product": "CCOC(=O)CC",
  "top_k": 5,
  "beam_size": 20,
  "return_images": true
}
```
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `product` | string | ✅ | — | Product SMILES |
| `top_k` | int | ❌ | `TOP_K_DEFAULT` (5) | Number of ranked predictions to return |
| `beam_size` | int | ❌ | `BEAM_SIZE` (20) | Beam width |
| `return_images` | bool | ❌ | `true` | Include base64 PNG |

**Success Response (200):**
```json
{
  "predictions": [
    {
      "rank": 1,
      "smiles": "CCO.CC(=O)O",
      "score": -0.234567,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgA...",
      "image_mime": "image/png"
    },
    ...
  ],
  "request_id": "retro_xyz789"
}
```

#### Error Response (400 / 422 / 500)
```json
{
  "error": "invalid_smiles",
  "detail": "RDKit could not parse reactants: 'CCO.CC(=O)O' — Parse error at position 5",
  "request_id": "fwd_abc123"
}
```

**Error Codes:**
| `error` | HTTP | Meaning |
|---|---|---|
| `invalid_smiles` | 400 | RDKit sanitization failed |
| `model_not_loaded` | 503 | Checkpoint not available |
| `inference_failed` | 500 | Model threw exception |
| `validation_error` | 422 | Pydantic schema violation |

---

### 2.4 Example curl Commands

```bash
# Health check
curl http://localhost:8000/health

# Forward prediction
curl -X POST http://localhost:8000/predict/forward \
  -H "Content-Type: application/json" \
  -d '{"reactants": "CCO.CC(=O)O", "top_k": 3}'

# Retrosynthesis
curl -X POST http://localhost:8000/predict/retro \
  -H "Content-Type: application/json" \
  -d '{"product": "CCOC(=O)CC", "top_k": 3}'

# Without images (smaller response)
curl -X POST http://localhost:8000/predict/forward \
  -H "Content-Type: application/json" \
  -d '{"reactants": "CCO.CC(=O)O", "return_images": false}'
```

---

### 2.5 Molecule Rendering Strategy

**Decision: Backend returns base64-encoded PNG via RDKit.**

Rationale:
- Zero frontend rendering dependencies (no SMILES Drawer, no Ketcher API for display)
- Consistent look (RDKit `Draw.MolToImage` with fixed size 300×300)
- Works offline once loaded; no CDN reliance for result images
- Base64 inline in JSON → single request, no CORS for image sub-requests

**Implementation (`rendering.py`):**
```python
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO

def smiles_to_base64_png(smiles: str, size: tuple = (300, 300)) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
```

- Forward: renders **product** molecule
- Retro: renders **precursor set** (multi-fragment; RDKit draws all fragments in one image)
- If `return_images=false`, omit `image_base64` / `image_mime` fields

---

### 2.6 Model Loading (Reuse US-006/007 Logic)

The backend **reuses** the proven inference logic from:
- `infer_forward.py` → `backend/models/forward.py`
- `infer_retro.py` → `backend/models/retro.py`
- `load_checkpoint.py` → `backend/models/loader.py`

Key adaptation points:
- Load **once at startup** (not per-request)
- Store model + vocab in app state (`app.state.forward_model`, `app.state.retro_model`)
- Force `device='cpu'` (configurable via `DEVICE` env)
- Use real vocabs: `dataset/USPTO_STEREO/vocab_smiles.txt` (406 tokens) and `dataset/USPTO_50k/vocab_smiles.txt` (75 tokens)
- RDKit-filter every candidate before returning (as in US-006/007)

---

## 3. Frontend Specification (Static, GitHub Pages)

### 3.1 Directory Structure (Frontend)

```
frontend/
├── index.html           # Single-page app (two tabs)
├── config.js            # Runtime config: window.BACKEND_URL
├── style.css            # Clean, responsive styling
├── app.js               # Tab logic, API calls, rendering
└── ketcher.html         # Optional: minimal Ketcher iframe page
```

### 3.2 `config.js` (Runtime Configuration)

```js
// Injected at build/deploy time or edited manually for local dev
window.BACKEND_URL = "http://localhost:8000";  // Dev default
// Production: window.BACKEND_URL = "https://api.example.com";
```

**Deployment (US-012):** GitHub Pages serves `frontend/` from `main`. `config.js` is replaced at deploy time with the production backend URL (via a simple `sed` in the deploy script).

---

### 3.3 UI Layout (Component Hierarchy)

```
index.html
├── <header>
│   └── <h1>RXNGraphormer Web</h1>
├── <nav class="tabs" role="tablist">
│   ├── <button role="tab" data-tab="forward" aria-selected="true">Forward Synthesis</button>
│   └── <button role="tab" data-tab="retro" aria-selected="false">Retrosynthesis</button>
├── <main>
│   ├── <section id="tab-forward" role="tabpanel" aria-labelledby="tab-forward-btn">
│   │   ├── <div class="input-area">
│   │   │   <label for="forward-smiles">Reactant SMILES</label>
│   │   │   <textarea id="forward-smiles" placeholder="CCO.CC(=O)O" rows="3"></textarea>
│   │   │   <div class="input-actions">
│   │   │     <button id="forward-draw" type="button">Draw in Ketcher</button>
│   │   │     <button id="forward-predict" type="button" class="primary">Predict</button>
│   │   │     <button id="forward-clear" type="button">Clear</button>
│   │   │   </div>
│   │   │   <div id="forward-error" class="error-message" aria-live="polite"></div>
│   │   │   </div>
│   │   ├── <div id="forward-loading" class="loading hidden" aria-live="polite">
│   │   │   <span class="spinner"></span> Running prediction...
│   │   │   </div>
│   │   └── <div id="forward-results" class="results-grid"></div>
│   │
│   └── <section id="tab-retro" role="tabpanel" aria-labelledby="tab-retro-btn" hidden>
│       ├── <div class="input-area">
│       │   <label for="retro-smiles">Product SMILES</label>
│       │   <textarea id="retro-smiles" placeholder="CCOC(=O)CC" rows="3"></textarea>
│       │   <div class="input-actions">
│       │     <button id="retro-draw" type="button">Draw in Ketcher</button>
│       │     <button id="retro-predict" type="button" class="primary">Predict</button>
│       │     <button id="retro-clear" type="button">Clear</button>
│       │   </div>
│       │   <div id="retro-error" class="error-message" aria-live="polite"></div>
│       │   </div>
│       ├── <div id="retro-loading" class="loading hidden" aria-live="polite">
│       │   <span class="spinner"></span> Running retrosynthesis...
│       │   </div>
│       └── <div id="retro-results" class="results-grid"></div>
└── <footer>
    <p>Powered by RXNGraphormer • <a href="https://github.com/milo0914/rxngraphormer-web">GitHub</a></p>
```

---

### 3.4 Result Card Component

Each prediction renders as a card:

```
┌─────────────────────────────────────────────────────────────┐
│  ┌─────┐  Rank 1  •  Score: -0.1235                         │
│  │     │                                                   │
│  │ IMG │  CCOC(=O)CC                                        │
│  │ 300×│  [Copy SMILES]  [Open in Ketcher]                  │
│  │ 300 │                                                   │
│  └─────┘                                                   │
└─────────────────────────────────────────────────────────────┘
```

**Card fields:**
- **Rank** (1..top_k)
- **Score** (beam log-prob, 4 decimal places)
- **Molecule image** (base64 PNG from backend, 300×300)
- **SMILES** (monospace, selectable)
- **Copy button** → copies SMILES to clipboard
- **Open in Ketcher button** → opens `https://ketcher.epam.com/ketcher?smiles=<urlencoded>`

---

### 3.5 Ketcher Integration

**Approach: External link (new tab) — simplest, no iframe complexity.**

- "Draw in Ketcher" button opens `https://ketcher.epam.com/ketcher` in a new tab
- User draws, copies SMILES from Ketcher, pastes into our textarea
- "Open in Ketcher" on result cards opens the predicted SMILES in Ketcher for inspection

**Alternative (if embed needed later):** `iframe` with `https://ketcher.epam.com/ketcher` + `postMessage` for SMILES exchange. Not in v1.

---

### 3.6 Error & Loading States

| State | UI Treatment |
|---|---|
| **Idle** | Empty results grid, enabled Predict button |
| **Loading** | Predict button disabled + spinner; "Running prediction..." toast; results grid hidden |
| **Success** | Results grid populated with cards; scroll to results |
| **Invalid SMILES (client-side)** | Inline red error under textarea: "Invalid SMILES: RDKit parse failed" (before API call, via simple heuristic: non-empty, no obvious garbage) |
| **API Error (400/500)** | Inline red error with `error` + `detail` from response; Predict button re-enabled |
| **Network Error** | "Cannot reach backend. Check BACKEND_URL in config.js" |
| **Empty Results** | "No valid predictions returned" (rare; model returned 0 RDKit-valid candidates) |

---

### 3.7 Responsiveness & Accessibility

- **Mobile-first CSS**: Stack tabs vertically on < 600px; result cards single-column
- **ARIA**: Proper `role="tablist"`, `aria-selected`, `aria-labelledby`, `aria-live` for loading/errors
- **Keyboard**: Tab navigation, Enter on Predict, Escape closes Ketcher link (handled by browser)
- **Color scheme**: Light mode only (v1); WCAG AA contrast
- **No JS framework** — vanilla ES6 modules (`type="module"` in script tag)

---

## 4. Checkpoint Fetching Strategy

### 4.1 Required Checkpoints

| Model | Local Path | Source (Priority) |
|---|---|---|
| Forward (USPTO_STEREO fine-tune) | `models/seq-v2-USPTO_STEREO-20250509_070206_ft/` | 1. GitHub Release `forward-stereo-ft` 2. figshare 59201306 |
| Retro (USPTO_50k) | `models/USPTO_50k/` | 1. GitHub Release `retro-uspto50k` 2. figshare 53998184 |
| Forward vocab | `dataset/USPTO_STEREO/vocab_smiles.txt` | Bundled in repo (committed, 406 tokens) |
| Retro vocab | `dataset/USPTO_50k/vocab_smiles.txt` | Bundled in repo (committed, 75 tokens) |

> **Vocabs are small (KB) and committed to the repo.** Only the large `.pt` checkpoints are fetched at runtime.

### 4.2 GitHub Release Assets (US-012)

When US-012 publishes releases, assets will be named:
- `forward-stereo-ft.7z` → extracts to `seq-v2-USPTO_STEREO-20250509_070206_ft/`
- `retro-uspto50k.7z` → extracts to `USPTO_50k/`

Download URLs (public repo):
```
https://github.com/milo0914/rxngraphormer-web/releases/download/v1.0.0/forward-stereo-ft.7z
https://github.com/milo0914/rxngraphormer-web/releases/download/v1.0.0/retro-uspto50k.7z
```

### 4.3 Figshare Fallback URLs

| Asset | Figshare Download URL |
|---|---|
| Forward fine-tune | `https://ndownloader.figshare.com/files/59201306` |
| Retro USPTO_50k | `https://ndownloader.figshare.com/files/53998184` |

### 4.4 `download_models.py` Specification

**Location:** `backend/download_models.py` (also usable standalone)

**Behavior:**
1. Check if required model dirs exist and contain `parameters.json` + `model/valid_checkpoint.pt`
2. If missing, download from GitHub Releases (using `GITHUB_RELEASES_REPO`, `GITHUB_TOKEN` if set)
3. If Release download fails (404, network), fall back to figshare URLs
4. Verify MD5 against known values (from `MODELS.md`)
5. Extract `.7z` / `.zip` using `py7zr` / `zipfile` (no system deps)
6. Write a `models/.manifest.json` recording source, version, MD5, timestamp
7. Exit 0 on success, non-zero on failure (blocks backend startup if `DOWNLOAD_MODELS_ON_START=true`)

**CLI:**
```bash
python backend/download_models.py              # Download all missing
python backend/download_models.py --forward-only
python backend/download_models.py --retro-only
python backend/download_models.py --verify     # Check existing only
```

---

## 5. Deployment

### 5.1 Backend (VPS Now → GCP Cloud Run Later)

**Dockerfile (multi-stage for small image):**
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY backend/ ./backend/
COPY dataset/ ./dataset/           # Vocabs (small, committed)
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["python", "-m", "backend.main"]
```

**`run.sh` (Local Dev):**
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /app/venv/bin/activate
export PYTHONPATH=/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web
python -m backend.main
```

### 5.2 Frontend (GitHub Pages)

- **Branch:** `main` (no Actions — PAT lacks `workflow` scope)
- **Source:** `/frontend` folder served from `main` branch
- **URL:** `https://milo0914.github.io/rxngraphormer-web/`
- **Deploy:** `scripts/deploy_pages.sh` (run locally, pushes `frontend/` to `main`)

### 5.3 CORS Configuration

Backend allows origin from `BACKEND_URL` (parsed for origin) + `https://milo0914.github.io` + `http://localhost:*` for dev.

```python
# config.py
BACKEND_URL = "http://localhost:8000"
ALLOWED_ORIGINS = [
    "http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:5500",
    "https://milo0914.github.io",
    BACKEND_URL,  # if backend serves frontend too
]
```

---

## 6. Development Workflow

### 6.1 Local Development

```bash
# Terminal 1: Backend
cd rxngraphormer-web
source /app/venv/bin/activate
export PYTHONPATH=$(pwd)
python -m backend.main
# → http://localhost:8000

# Terminal 2: Frontend (simple HTTP server)
cd rxngraphormer-web/frontend
python -m http.server 8080
# → http://localhost:8080 (edit config.js to point to http://localhost:8000)
```

### 6.2 Adding New Checkpoints

1. Upload `.7z`/`.zip` to GitHub Release (US-012)
2. Update `download_models.py` with new asset names + MD5s
3. Update `MODELS.md` table
4. Bump version tag

---

## 7. Simplicity & Ease-of-Use Checklist

| Principle | Implementation |
|---|---|
| **No build step** | Frontend = static HTML/JS/CSS; no npm, webpack, vite |
| **Single HTML file** | `index.html` loads `config.js`, `style.css`, `app.js` as separate files (cacheable) but could be inlined |
| **Configurable backend** | `config.js` + env vars; no code changes to switch VPS ↔ GCP |
| **Self-contained backend** | Dockerfile installs all deps; `download_models.py` fetches weights |
| **Minimal deps** | Backend: fastapi, uvicorn, pydantic, pydantic-settings, rdkit, torch, torch-geometric, py7zr, python-box, OpenNMT-py, tqdm |
| **Clear errors** | Structured JSON errors with `error` code + human `detail` |
| **Copy-paste ready** | Copy SMILES buttons; Ketcher links for visual editing |
| **Documented API** | Example curl commands in DESIGN.md + `/docs` (FastAPI Swagger UI) |

---

## 8. File Summary (What US-009 Delivers)

| File | Purpose |
|---|---|
| `DESIGN.md` | This document — single source of truth for US-010/011/012 |
| `backend/download_models.py` | Startup model fetcher (GitHub Releases → figshare fallback) |
| `backend/requirements.txt` | Backend Python dependencies |
| `frontend/config.js` | Runtime backend URL configuration |
| (Future US-010) `backend/main.py`, `backend/config.py`, `backend/models/...`, `backend/schemas.py`, `backend/rendering.py`, `backend/Dockerfile`, `backend/run.sh` |
| (Future US-011) `frontend/index.html`, `frontend/style.css`, `frontend/app.js` |

---

## 9. Open Decisions (Resolved in This Spec)

| Question | Decision |
|---|---|
| Molecule rendering: backend vs frontend? | **Backend** returns base64 PNG (RDKit) |
| Ketcher: embed iframe or external link? | **External link** (new tab) — simpler, no postMessage complexity |
| Frontend build step? | **None** — static files on GitHub Pages |
| Config injection for frontend? | **`config.js`** with `window.BACKEND_URL` (replaced at deploy) |
| Checkpoint fetch: when? | **Backend startup** (optional, controlled by `DOWNLOAD_MODELS_ON_START`) |
| Vocabs in repo or fetched? | **In repo** (committed, small) — only `.pt` weights fetched |

---

## 10. References

- `MODELS.md` — Checkpoint sources, MD5s, figshare IDs
- `PROJECT_NOTES.md` — Framework entry points, vocab loading quirks
- `infer_forward.py` / `infer_retro.py` — Proven inference logic (CPU, RDKit-filtered)
- `REPRODUCTION_REPORT.md` — Accuracy benchmarks (forward 76% top-1, retro 50% top-1)
- US-006/007 commits — Working inference implementations to wrap

---

*End of DESIGN.md — ready for US-010 (Backend) and US-011 (Frontend) implementation.*