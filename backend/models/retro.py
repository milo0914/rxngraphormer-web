"""Retrosynthesis prediction API wrapper."""
from typing import List, Tuple

from .loader import retro_predict
from backend.rendering import is_valid_smiles


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
    """
    # Run inference
    predictions = retro_predict(product, top_k, beam_size, temperature, device)
    
    # Filter for RDKit-valid SMILES, preserving score order
    valid = [(smi, score) for smi, score in predictions if is_valid_smiles(smi)]
    
    return valid[:top_k]