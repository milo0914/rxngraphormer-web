"""Forward prediction API wrapper."""
from typing import List, Tuple

from .loader import forward_predict
from backend.rendering import is_valid_smiles


def predict_forward(
    reactants: str,
    top_k: int = 5,
    beam_size: int = 20,
    temperature: float = 1.0,
    device: str = "cpu",
) -> List[Tuple[str, float]]:
    """
    Run forward prediction and filter for valid SMILES.
    
    Returns:
        List of (product_smiles, score) tuples, score-ordered, RDKit-valid only.
    """
    # Run inference
    predictions = forward_predict(reactants, top_k, beam_size, temperature, device)
    
    # Filter for RDKit-valid SMILES, preserving score order
    valid = [(smi, score) for smi, score in predictions if is_valid_smiles(smi)]
    
    return valid[:top_k]