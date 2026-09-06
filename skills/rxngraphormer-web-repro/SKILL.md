---
description: 'Use this skill when you want to run RXNGraphormer Web locally on this
  VPS: start backend (port 8000) + frontend (port 8080), run forward/retrosynthesis
  prediction verification via API, and display frontend interface summary. Triggers:
  "run rxngraphormer repro", "start rxngraphormer locally", "verify rxngraphormer
  predictions", "rxngraphormer web demo", "啟動 rxngraphormer", "跑驗證指令".'
name: rxngraphormer-web-repro
---

# RXNGraphormer Web Local Reproduction & Verification Skill

This skill starts the RXNGraphormer Web backend + frontend on this VPS, runs verification commands for both forward prediction and retrosynthesis, and displays frontend interface information.

## Execution

This skill ships with a batch JSON file `scripts/rxngraphormer-repro.json`.

**Call `run_tool_batch` strictly in the format below, using `file_path` to load the file. Do not construct your own `actions` list inline.**

`run_tool_batch` requires an **absolute path** for `file_path`. Use the absolute directory path you see when reading this SKILL.md to construct the full path.

```
run_tool_batch(
  file_path="<this skill dir>/scripts/rxngraphormer-repro.json",
  args={}
)
```

### Batch Parameters

This batch uses no runtime parameters (`args={}`). All paths are hardcoded to the project location at `/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/`.

### Batch failure handling

If `run_tool_batch` fails (returns `ok: false` or errors mid-way):
1. Check the error message — common issues: backend already running on port 8000, frontend already on 8080, or model weights not downloaded.
2. If ports are in use, kill existing processes: `kill $(cat /tmp/backend.pid 2>/dev/null) $(cat /tmp/frontend.pid 2>/dev/null) 2>/dev/null; pkill -f "uvicorn main:app" 2>/dev/null; pkill -f "http.server 8080" 2>/dev/null`.
3. If model weights missing, run `python backend/download_models.py --verify` first.
4. Re-run the batch.
5. If still fails, fall back to manual step-by-step execution using the "Step-by-step reference" section below.

### Step-by-step reference

The following details each batch step, for debugging or manual execution only:

1. **Environment check** — Verifies repo structure and Python environment (torch CPU, rdkit). Runs from repo root.
2. **Start backend** — Launches FastAPI via uvicorn on 127.0.0.1:8000, waits 8s, verifies `/health` endpoint returns `{"status":"ok",...}`.
3. **Start frontend** — Launches Python http.server on 127.0.0.1:8080 serving `frontend/`, waits 3s, verifies HTML loads.
4. **Forward prediction verification** — POSTs `CCO.CC(=O)O` (ethanol + acetic acid) to `/predict/forward` with `top_k=3, return_images=true`. Validates 3 predictions returned with valid SMILES, scores, and base64 PNG images.
5. **Retrosynthesis verification** — POSTs `CCOC(=O)CC` (ethyl acetate) to `/predict/retro` with `top_k=3, return_images=true`. Validates 3 precursor sets returned with valid SMILES, scores, and base64 PNG images.
6. **Frontend interface summary** — Prints access URLs, frontend tab names (Forward Synthesis, Retrosynthesis), and lists all API endpoints from OpenAPI spec.

### Notes

- **Ports**: Backend 8000, Frontend 8080. Both bind to 127.0.0.1 (local only).
- **Process management**: PIDs saved to `/tmp/backend.pid` and `/tmp/frontend.pid`. Logs at `/tmp/backend.log`, `/tmp/frontend.log`.
- **Model weights**: Must exist in `models/` (downloaded by US-004/US-007). If missing, run `python backend/download_models.py --verify` first.
- **Cleanup**: The batch does NOT stop servers after verification. To stop: `kill $(cat /tmp/backend.pid) $(cat /tmp/frontend.pid) 2>/dev/null`.
- **Frontend config**: `frontend/config.js` has `BACKEND_URL = "http://localhost:8000"` — matches local backend.
- **GPU**: Runs on CPU only (`torch 2.1.2+cpu`). No GPU required.

### Worked example

```
run_tool_batch(
  file_path="/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/.qwenpaw/skills/rxngraphormer-web-repro/scripts/rxngraphormer-repro.json",
  args={}
)
```

Expected output (abridged):
```
=== 1. Environment check ===
torch: 2.1.2+cpu cuda: False
rdkit ok
=== 2. Backend started ===
{"status":"ok","forward_model_loaded":true,"retro_model_loaded":true,"device":"cpu"}
=== 3. Frontend started ===
<!DOCTYPE html>...
=== 4. Forward prediction verification ===
predictions: 3
  1: O=C1c2ccccc2C(=O)C1(O)O (score=-0.1234) img=True
  2: ...
=== 5. Retrosynthesis prediction verification ===
predictions: 3
  1: CC(=O)Cl.O=C(O)c1ccccc1O (score=-0.0002) img=True
  ...
=== 6. Frontend interface summary ===
Frontend URL: http://localhost:8080
Backend URL:  http://localhost:8000
Frontend tabs: Forward Synthesis, Retrosynthesis
API endpoints: GET /health, POST /predict/forward, POST /predict/retro
Verification complete.
```

### Failure modes and recovery

| Symptom | Cause | Recovery |
|---------|-------|----------|
| `curl: (7) Failed to connect` | Backend not started or crashed | Check `/tmp/backend.log`; ensure models exist; restart step 2 |
| `{"detail":"Forward model not loaded"}` | Model weights missing or load failed | Run `python backend/download_models.py --verify` then restart backend |
| Port 8000/8080 already in use | Previous run didn't clean up | `kill $(cat /tmp/backend.pid) $(cat /tmp/frontend.pid) 2>/dev/null` then retry |
| `return_images=true` but `image_base64` missing | RDKit render failed | Check RDKit install; model may return invalid SMILES that can't render |

---

## Model Backup & Recovery

### Model Files Location

Model weights and datasets are stored outside the default agent workspace to keep backup sizes manageable. After any workspace backup/restore, models must be restored separately.

**Current location** (after 2026-09-05 workspace cleanup):
```
/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/
└── backups_20260905_rxngraphormer/     ← rxngraphormer-web 完整目錄（5.1 GB）
    ├── models/                           ← 模型權重（1.9 GB）
    │   ├── seq-v2-USPTO_STEREO-20250509_070206_ft/  Forward checkpoint（最新）
    │   ├── seq-v2-USPTO_STEREO-20250423_044122_ft/  Forward checkpoint（基礎版）
    │   └── USPTO_50k/                               Retrosynthesis checkpoint
    ├── data/                            ← 資料集（1.9 GB）
    │   ├── USPTO_STEREO/
    │   └── USPTO_50k/
    ├── rxngraphormer/                   ← 模型框架原始碼（1.0 GB）
    ├── download_models.sh                 ← 一鍵復原腳本
    ├── MODELS.md                         ← 模型說明文件（詳細 MD5/來源）
    └── backend/frontend/docs/            ← 部署腳本與文件
```

### Model Inventory

| 模型 | 任務 | 大小 | 來源 |
|------|------|------|------|
| `seq-v2-USPTO_STEREO-20250509_070206_ft/` | **Forward prediction**（最新） | ~374 MB | figshare 59201306 |
| `seq-v2-USPTO_STEREO-20250423_044122_ft/` | Forward prediction（基礎版） | ~374 MB | figshare 59201303 |
| `USPTO_50k/` | **Retrosynthesis** | ~189 MB | figshare 53998184 |

MD5 均已驗證，詳見 `MODELS.md`。

### Recovery Methods

#### Method A: Script one-liner（推薦）

The `download_models.sh` script auto-downloads from GitHub Releases with MD5 verification and resume support:

```bash
cd /路徑/to/rxngraphormer-web
bash download_models.sh              # 下載全部 checkpoint
bash download_models.sh --verify    # 僅驗證，不下載
bash download_models.sh --forward-only  # 只下載 forward 模型
bash download_models.sh --retro-only   # 只下載 retro 模型
```

**Primary source**: GitHub Releases → `https://github.com/milo0914/rxngraphormer-web/releases/tag/v1.0-weights`
**Fallback**: figshare（`download_models.sh` 自動切換）

#### Method B: Restore from local backup（最快）

If `backups_20260905_rxngraphormer/` still exists, move it back directly:

```bash
mv /run/csi/mount-root/nas/.../backups_20260905_rxngraphormer/rxngraphormer-web \
   /run/csi/mount-root/nas/.../workspaces/default/rxngraphormer-web
```

#### Method C: Skill-driven（agent 可執行）

Agent can invoke the `rxngraphormer-web-repro` skill after models are restored to restart backend + frontend.

### Download Sources

| 來源 | 用途 | 網址 |
|------|------|------|
| **GitHub Releases**（主要） | `forward-stereo-ft.7z` / `retro-uspto50k.7z` | https://github.com/milo0914/rxngraphormer-web/releases/tag/v1.0-weights |
| **figshare**（備援） | 各別模型 `.7z` 檔 | https://figshare.com/articles/30498368（Forward）<br>https://figshare.com/articles/28356077（Retro） |

Both sources verified with MD5. Script supports resume (`curl -C -`).

## Prerequisites

- Project repo at `/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web/`
- Python venv at `/app/venv` with torch 2.1.2+cpu, rdkit-pypi, torch-geometric, OpenNMT-py
- Model weights in `models/` (forward: `seq-v2-USPTO_STEREO-20250509_070206_ft/`, retro: `USPTO_50k/`)
- Vocab files in `dataset/USPTO_STEREO/vocab_smiles.txt` and `dataset/USPTO_50k/vocab_smiles.txt`

## Edge cases

- Running multiple times without cleanup leaves orphan processes — always clean up before re-run.
- First backend start downloads nothing (weights already present) but loads models into memory (~2-3s).
- Retrosynthesis model has ~12% invalid SMILES rate (known limitation, see REPRODUCTION_REPORT.md).