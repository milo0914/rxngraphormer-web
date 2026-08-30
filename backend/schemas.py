"""Pydantic request/response models for the API."""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ErrorCode(str, Enum):
    INVALID_SMILES = "invalid_smiles"
    MODEL_NOT_LOADED = "model_not_loaded"
    INFERENCE_FAILED = "inference_failed"
    VALIDATION_ERROR = "validation_error"


class HealthResponse(BaseModel):
    status: str = "ok"
    forward_model_loaded: bool
    retro_model_loaded: bool
    device: str


class ForwardRequest(BaseModel):
    reactants: str = Field(..., description="Reactant SMILES (dot-separated for multiple)")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of ranked predictions to return")
    beam_size: Optional[int] = Field(None, ge=1, le=50, description="Beam width for search")
    return_images: bool = Field(True, description="Include base64 PNG in response")


class RetroRequest(BaseModel):
    product: str = Field(..., description="Product SMILES")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of ranked predictions to return")
    beam_size: Optional[int] = Field(None, ge=1, le=50, description="Beam width for search")
    return_images: bool = Field(True, description="Include base64 PNG in response")


class PredictionItem(BaseModel):
    rank: int
    smiles: str
    score: float
    image_base64: Optional[str] = None
    image_mime: Optional[str] = None


class ForwardResponse(BaseModel):
    predictions: List[PredictionItem]
    request_id: str


class RetroResponse(BaseModel):
    predictions: List[PredictionItem]
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorCode
    detail: str
    request_id: Optional[str] = None