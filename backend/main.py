"""FastAPI backend for RXNGraphormer prediction service."""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.schemas import (
    ErrorCode,
    ErrorResponse,
    ForwardRequest,
    ForwardResponse,
    HealthResponse,
    RetroRequest,
    RetroResponse,
)
from backend.models.loader import (
    load_forward_model,
    load_retro_model,
    is_forward_loaded,
    is_retro_loaded,
)
from backend.models.forward import predict_forward
from backend.models.retro import predict_retro
from backend.rendering import smiles_to_base64_png, is_valid_smiles


# ─── Lifespan (startup/shutdown) ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup."""
    # Startup
    print("=" * 60)
    print("Starting RXNGraphormer Backend...")
    print(f"Device: {settings.device}")
    print(f"Forward model dir: {settings.forward_model_dir}")
    print(f"Retro model dir: {settings.retro_model_dir}")
    print("=" * 60)
    
    # Download models if configured
    if settings.download_models_on_start:
        print("Downloading models...")
        import subprocess
        result = subprocess.run(
            ["python", "backend/download_models.py"],
            cwd="/run/csi/mount-root/nas/4079184d856ecc166ed19d4887083405/workspaces/default/rxngraphormer-web",
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: Model download failed: {result.stderr}")
        else:
            print("Models downloaded successfully")
    
    # Load forward model
    try:
        load_forward_model(
            settings.forward_model_dir,
            settings.forward_vocab_path,
            settings.device,
        )
        print("Forward model loaded")
    except Exception as e:
        print(f"Warning: Failed to load forward model: {e}")
    
    # Load retro model
    try:
        load_retro_model(
            settings.retro_model_dir,
            settings.retro_vocab_path,
            settings.device,
        )
        print("Retro model loaded")
    except Exception as e:
        print(f"Warning: Failed to load retro model: {e}")
    
    print("Backend ready!")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("Shutting down...")


# ─── FastAPI App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="RXNGraphormer Prediction API",
    description="Forward synthesis and retrosynthesis prediction service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Exception Handlers ───────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    if exc.status_code == 400:
        error = ErrorCode.INVALID_SMILES
    elif exc.status_code == 503:
        error = ErrorCode.MODEL_NOT_LOADED
    else:
        error = ErrorCode.INFERENCE_FAILED
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=error, detail=exc.detail, request_id=request_id).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorCode.INFERENCE_FAILED,
            detail=f"Internal server error: {str(exc)}",
            request_id=request_id,
        ).model_dump(),
    )


# ─── Middleware: Request ID ───────────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ─── Health Check ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        forward_model_loaded=is_forward_loaded(),
        retro_model_loaded=is_retro_loaded(),
        device=settings.device,
    )


# ─── Validation Helper ────────────────────────────────────────────────────

def validate_smiles_or_400(smiles: str, field_name: str) -> None:
    """Validate SMILES with RDKit, raise 400 if invalid."""
    if not is_valid_smiles(smiles):
        raise HTTPException(
            status_code=400,
            detail=f"RDKit could not parse {field_name}: '{smiles}' — invalid SMILES",
        )


# ─── Forward Prediction Endpoint ──────────────────────────────────────────

@app.post("/predict/forward", response_model=ForwardResponse)
async def predict_forward_endpoint(request: ForwardRequest, req: Request):
    request_id = req.state.request_id
    
    # Validate input SMILES
    validate_smiles_or_400(request.reactants, "reactants")
    
    # Check model loaded
    if not is_forward_loaded():
        raise HTTPException(
            status_code=503,
            detail="Forward model not loaded",
        )
    
    # Use defaults from settings if not provided
    top_k = request.top_k or settings.top_k_default
    beam_size = request.beam_size or settings.beam_size
    
    try:
        # Run prediction
        predictions = predict_forward(
            request.reactants,
            top_k=top_k,
            beam_size=beam_size,
            device=settings.device,
        )
        
        # Build response
        pred_items = []
        for rank, (smi, score) in enumerate(predictions, start=1):
            item = {
                "rank": rank,
                "smiles": smi,
                "score": score,
            }
            if request.return_images:
                img_b64 = smiles_to_base64_png(smi)
                if img_b64:
                    item["image_base64"] = img_b64
                    item["image_mime"] = "image/png"
            pred_items.append(item)
        
        return ForwardResponse(predictions=pred_items, request_id=request_id)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )


# ─── Retrosynthesis Endpoint ──────────────────────────────────────────────

@app.post("/predict/retro", response_model=RetroResponse)
async def predict_retro_endpoint(request: RetroRequest, req: Request):
    request_id = req.state.request_id
    
    # Validate input SMILES
    validate_smiles_or_400(request.product, "product")
    
    # Check model loaded
    if not is_retro_loaded():
        raise HTTPException(
            status_code=503,
            detail="Retrosynthesis model not loaded",
        )
    
    # Use defaults from settings if not provided
    top_k = request.top_k or settings.top_k_default
    beam_size = request.beam_size or settings.beam_size
    
    try:
        # Run prediction
        predictions = predict_retro(
            request.product,
            top_k=top_k,
            beam_size=beam_size,
            device=settings.device,
        )
        
        # Build response
        pred_items = []
        for rank, (smi, score) in enumerate(predictions, start=1):
            item = {
                "rank": rank,
                "smiles": smi,
                "score": score,
            }
            if request.return_images:
                img_b64 = smiles_to_base64_png(smi)
                if img_b64:
                    item["image_base64"] = img_b64
                    item["image_mime"] = "image/png"
            pred_items.append(item)
        
        return RetroResponse(predictions=pred_items, request_id=request_id)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )


# ─── Main Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
    )