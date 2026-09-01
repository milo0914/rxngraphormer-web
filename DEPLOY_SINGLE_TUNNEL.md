# 單通道部署方案（Single-Tunnel Deployment）

> 本文件記錄 2026-09-01 的單一 ngrok tunnel + 單一 VPS 程序部署實驗與心得。

---

## 1. 架構決策

### 原始問題
ngrok 免費版一個 authtoken 只能開**一條** tunnel。若堅持「前端、後端分開程序」架構，需要兩條 tunnel → 需要兩個 token 或付費方案。

### 解決方案：單一入口
讓 **backend 同時擔任靜態檔案伺服器**，所有靜態資源（index.html、app.js、style.css）由後端在 `/static/` 路徑下提供，API 維持在根路徑 `/predict/*` 與 `/health`。前端 UI 與後端 API 共用同一個 origin，瀏覽器直接以相對路徑呼叫 API，完全繞過跨域問題。

```
互聯網用戶
    │
    │ https://xxx.ngrok-free.dev/   ← 單一 ngrok tunnel (8001)
    ▼
ngrok Edge (port 80/443)
    │
    ▼
VPS: uvicorn (port 8001)
    ├── GET /              → index.html（前端 UI）
    ├── GET /static/*      → 靜態資產（app.js, style.css, config.js）
    ├── GET /health        → 模型狀態
    ├── POST /predict/forward → 前向合成預測
    └── POST /predict/retro  → 逆合成預測
```

### 為何只需一條 tunnel
前端和後端都在同一個 backend 程序（同一 port），tunnel 綁定 port 8001，互聯網用戶的瀏覽器從同一 URL 取得 UI 頁面再以相對路徑呼叫 API，自然解決 CORS 問題。

---

## 2. 實作修改

### 2.1 `backend/main.py`（新增）

```python
# ─── Serve frontend static files (single-origin deployment) ────────────────
# Mount at /static so that API routes at / (health, predict, etc.) are NOT
# intercepted.  A manual GET / returns index.html so the root URL works.
import os as _os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_FE_DIR = _os.path.join(
    _os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend"
)
if _os.path.isdir(_FE_DIR):
    app.mount("/static", StaticFiles(directory=_FE_DIR, html=True), name="frontend")

    @app.get("/")
    async def serve_index():
        return FileResponse(_os.path.join(_FE_DIR, "index.html"))
```

**關鍵設計**：`StaticFiles` 掛在 `/static` 而非 `/`。若掛在 `/`，會攔截所有路由（包含 `/health`、`/predict/*`），導致 API 全部回 404。

### 2.2 `frontend/index.html`（修改 `<head>` 區塊）

```html
<!-- 單一 origin：直接用當前頁面的 origin，無需 external config -->
<script>
  window.BACKEND_URL = window.location.origin || "";
</script>
```

**`config.js` 的教訓**：`config.js` 設 `BACKEND_URL = ""`（空字串），但 `app.js` 原始邏輯是：

```javascript
const BACKEND_URL = (typeof window.BACKEND_URL === "string" && window.BACKEND_URL.trim())
    ? window.BACKEND_URL.trim()
    : "http://localhost:8000";  // ← 空字串是 falsy，fallback 到 hardcoded！
```

空字串 `.trim()` 後仍為 `""`（falsy），導致瀏覽器 fallback 到 `http://localhost:8000`，即使 `config.js` 有設定也不行。**解決方案**：使用 `window.location.origin`（永遠非空）或確保 `config.js` 裡 `BACKEND_URL` 永遠是完整 URL 而非空字串。

### 2.3 `frontend/index.html`（資源路徑更新）

```html
<link rel="stylesheet" href="/static/style.css">
<script src="/static/config.js"></script>
<script type="module" src="/static/app.js"></script>
```

---

## 3. 啟動指令（單 VPS 程序）

```bash
# 殺掉舊程序
pkill -f "uvicorn backend.main:app" || true

# 啟動單一 backend（含前端 + API）
cd /path/to/rxngraphormer-web
/app/venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001

# 另一 terminal：設定 ngrok token（如尚未設定）
ngrok config add-authtoken YOUR_TOKEN

# 啟動 tunnel → 單一通道暴露 port 8001
ngrok http 8001
```

---

## 4. ngrok 免費版限制驗證

| 場景 | 是否需要兩個 token | 說明 |
|------|-------------------|------|
| 前後端分開程序（前端 8080 + 後端 8000） | ✅ 需要（分開 tunnel） | 每個 tunnel 需要一個 token |
| 單一入口（後端 serving 靜態，port 8001） | ❌ 一個就夠 | 同一程序同時提供 UI + API |
| 付费版 ngrok（支援多隧道） | ❌ 不需要 | 付費版支援同一 token 多 tunnel |

---

## 5. 常見錯誤與排除

### 錯誤 1：`Cannot reach backend at http://localhost:8000`
**原因**：`config.js` 設 `BACKEND_URL = ""`，`app.js` fallback 到 hardcoded `localhost:8000`。
**修復**：確保 `window.BACKEND_URL` 不為空字串。用 `window.location.origin`（推薦）或填入完整 URL。

### 錯誤 2：`StaticFiles 掛在 / 導致 API 全 404`
**原因**：FastAPI 的 `StaticFiles` mount 在 `/` 會優先攔截所有 HTTP request。
**修復**：mount 到 `/static`，手動新增 `GET /` 回傳 `index.html`。

### 錯誤 3：`ngrok tunnel is offline (ERR_NGROK_3200)`
**原因**：後端程序崩潰（常見於 RDKit 解析無效分子時記憶體錯誤或 oom）。
**修復**：檢查 backend log，重新啟動後端。

### 錯誤 4：`Invalid SMILES: RDKit could not parse`
**原因**：前端傳入的 SMILES 字串語法有誤，非 RDKit 分子表示法。
**修復**：使用 ChemDraw、PubChem 或 RDKit 驗證分子式是否正確。

---

## 6. 關於 RDKit 與 USPTO 資料集覆蓋度

測試 SMILES `CCCOc3ccc2c1cc(OCC)sc1cc(F)cc2c3F`（含多個芳香環與雜原子）在 RDKit 解析時失敗：`Ring closure 1 used in two different ring closures`。這可能代表：
- SMILES 語法本身有問題
- 或 USPTO-50k / USPTO-STEREO 資料集對此結構的覆蓋不足

如需處理含大量芳香環或雜原子（F、Cl、S）的分子，建議另找覆蓋度更高的化學模型。

---

## 7. 未解決的技術債

- ⚠️ ngrok 免費版 tunnel 數小時後會自動斷線，需要重新啟動（適合開發驗證，不適合正式 production）
- ⚠️ `backend/main.py` 中 `CORS allow_origins=["*"]` 設為全開放，線上環境應限制為 ngrok 域名
- ⚠️ 模型長時間運行後記憶體累積問題尚待觀察
