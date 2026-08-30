# RXNGraphormer Web

Easy web UI for RXNGraphormer forward & retrosynthesis prediction.

---

## Quick Start (Local)

### 1. Backend (FastAPI)

```bash
# Install dependencies (CPU-only torch)
cd backend
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.1.2+cpu.html

# Run the server (downloads models on first start)
python -m backend.main
# Server runs at http://localhost:8000
```

The backend will automatically download model checkpoints from GitHub Releases (with figshare fallback) on first run.
Models are stored in `models/` (gitignored).

### 2. Frontend (Static)

```bash
# From repo root
cd frontend
python -m http.server 8080
# Open http://localhost:8080
```

The frontend reads `window.BACKEND_URL` from `frontend/config.js` (default: `http://localhost:8000`).

---

## Docker Deployment (VPS / GCP Cloud Run)

### Build the backend image

```bash
docker build -t rxngraphormer-backend ./backend
```

### Run locally with Docker

```bash
docker run -d \
  -p 8000:8000 \
  -e GITHUB_TOKEN=<your_github_token> \
  -e DOWNLOAD_MODELS_ON_START=true \
  --name rxngraphormer-backend \
  rxngraphormer-backend
```

### Deploy to GCP Cloud Run (future)

```bash
# Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/<PROJECT_ID>/rxngraphormer-backend ./backend

# Deploy to Cloud Run
gcloud run deploy rxngraphormer-backend \
  --image gcr.io/<PROJECT_ID>/rxngraphormer-backend \
  --platform managed \
  --region <REGION> \
  --allow-unauthenticated \
  --set-env-vars GITHUB_TOKEN=<token>,DOWNLOAD_MODELS_ON_START=true
```

> **Note:** The Dockerfile uses a multi-stage build. Model checkpoints are NOT baked into the image — they are downloaded at container startup via `download_models.py` (triggered by FastAPI lifespan). Set `DOWNLOAD_MODELS_ON_START=true` in production.

---

## GitHub Pages Demo

The frontend is deployed to **GitHub Pages**:

🔗 **https://milo0914.github.io/rxngraphormer-web/**

> By default, the Pages demo points to `http://localhost:8000` for the backend.
> To use the demo with a hosted backend, update `frontend/config.js` (or `docs/config.js`) with your backend URL and re-deploy.

### Deploy frontend to GitHub Pages manually

```bash
# The docs/ folder is the Pages source (synced from frontend/)
cp -r frontend/* docs/
git add docs/
git commit -m "Update GitHub Pages frontend"
git push origin main
# GitHub Pages auto-builds from main:/docs
```

---

## Model Checkpoints

Pre-trained checkpoints are hosted on **GitHub Releases**:

🔗 **https://github.com/milo0914/rxngraphormer-web/releases/tag/v1.0-weights**

| Model | Task | Size | MD5 |
|-------|------|------|-----|
| `forward-stereo-ft.7z` | Forward synthesis (USPTO_STEREO) | 391 MB | `52d506a2ecee0c77cad7de03c692f653` |
| `retro-uspto50k.7z` | Retrosynthesis (USPTO_50k) | 197 MB | `1d993b40b8ff38def31788c1ced69de5` |

### Download models manually

```bash
# Using the Python script (recommended)
python backend/download_models.py

# Using the shell script
bash download_models.sh

# Options:
#   --forward-only    Only download forward model
#   --retro-only      Only download retro model
#   --verify          Verify existing models only
#   --force           Re-download even if present
```

If GitHub Releases is unavailable, the scripts automatically fall back to figshare.

See [MODELS.md](MODELS.md) for detailed checkpoint information.

---

## Configuration

### Frontend → Backend URL

Edit `frontend/config.js` (or `docs/config.js` for Pages):

```javascript
window.BACKEND_URL = "https://your-backend.example.com";
// Default: "http://localhost:8000"
```

### Backend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub PAT for Releases API (rate limits) | — |
| `GITHUB_RELEASES_REPO` | Repo for model releases | `milo0914/rxngraphormer-web` |
| `DOWNLOAD_MODELS_ON_START` | Download models at startup | `true` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `http://localhost:8080,https://milo0914.github.io` |

---

## API Reference

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "ok",
  "forward_model_loaded": true,
  "retro_model_loaded": true,
  "device": "cpu"
}
```

### Forward Synthesis

```
POST /predict/forward
Content-Type: application/json

{
  "reactants": "CCO.CC(=O)O",
  "top_k": 5,
  "beam_size": 10,
  "return_images": true
}
```

Response:
```json
{
  "predictions": [
    {
      "rank": 1,
      "smiles": "CCOC(=O)CC",
      "score": -0.123,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

### Retrosynthesis

```
POST /predict/retro
Content-Type: application/json

{
  "product": "CCOC(=O)CC",
  "top_k": 5,
  "beam_size": 10,
  "return_images": true
}
```

Response:
```json
{
  "predictions": [
    {
      "rank": 1,
      "smiles": "CCO.CC(=O)O",
      "score": -0.456,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

### Error Responses

```json
{
  "error": "invalid_smiles",
  "detail": "RDKit could not parse reactants: INVALID"
}
```

Error codes: `invalid_smiles`, `model_not_loaded`, `inference_failed`.

---

## Project Structure

```
rxngraphormer-web/
├── backend/                 # FastAPI backend
│   ├── main.py             # FastAPI app + lifespan
│   ├── config.py           # Pydantic settings
│   ├── schemas.py          # Request/response models
│   ├── rendering.py        # RDKit molecule → base64 PNG
│   ├── download_models.py  # Model fetcher (GitHub Releases → figshare)
│   ├── models/             # Model loading & inference
│   │   ├── loader.py       # Core model loading
│   │   ├── forward.py      # Forward prediction wrapper
│   │   └── retro.py        # Retro prediction wrapper
│   ├── Dockerfile          # Multi-stage Docker build
│   ├── run.sh              # Local dev runner
│   └── requirements.txt    # Python dependencies
├── frontend/               # Static frontend (served locally)
│   ├── index.html          # Main HTML
│   ├── app.js              # Vanilla ES6 app
│   ├── style.css           # Styling
│   └── config.js           # Runtime BACKEND_URL
├── docs/                   # GitHub Pages source (copy of frontend/)
├── rxngraphormer/          # Vendored RXNGraphormer framework (US-003)
├── dataset/                # Vocabularies (committed, small)
│   ├── USPTO_STEREO/vocab_smiles.txt
│   └── USPTO_50k/vocab_smiles.txt
├── models/                 # Checkpoints (gitignored, downloaded at runtime)
├── download_models.sh      # Shell script for model downloads
├── MODELS.md               # Checkpoint documentation
├── DESIGN.md               # Architecture & API design (US-009)
├── PROJECT_NOTES.md        # Development notes
├── REPRODUCTION_REPORT.md  # Paper reproduction results (US-008)
└── requirements.txt        # Root requirements (base ML env)
```

---

## License

This project reproduces RXNGraphormer (MIT License). The web UI code is MIT licensed.
Model checkpoints are from the original authors (figshare articles 30498368, 28356077).