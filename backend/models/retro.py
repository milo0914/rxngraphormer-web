"""Retrosynthesis prediction API wrapper."""
import logging
from typing import List, Tuple

from .loader import retro_predict
from backend.rendering import is_valid_smiles

logger = logging.getLogger(__name__)


def predict_retro(
    product: str,
    top_k: int = 5,
    beam_size: int = 20,
    temperature: float = 1.0,
    device: str = "cpu",
) -> List[Tuple[str, float]]:
    """
    Run retrosynthesis prediction and filter for valid SMILES.
    
    Returns:
        List of (precursor_set_smiles, score) tuples, score-ordered, RDKit-valid only.
        Length is at most top_k. Logs warning if fewer than top_k valid results.
    """
    # Run inference (loader.py generates extra candidates internally)
    predictions = retro_predict(product, top_k, beam_size, temperature, device)
    
    # Filter for RDKit-valid SMILES, preserving score order
    valid = [(smi, score) for smi, score in predictions if is_valid_smiles(smi)]
    valid = valid[:top_k]
    
    if len(valid) < top_k:
        logger.warning(
            "Retro prediction: requested top_k=%d but only %d valid SMILES found (from %d candidates)",
            top_k, len(valid), len(predictions)
        )
    
    return valid