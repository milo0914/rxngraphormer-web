"""Molecule rendering utilities using RDKit."""
import base64
from io import BytesIO
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Draw


def smiles_to_base64_png(smiles: str, size: Tuple[int, int] = (300, 300)) -> Optional[str]:
    """
    Convert SMILES to base64-encoded PNG image.
    
    Args:
        smiles: SMILES string (can be multi-fragment for precursor sets)
        size: Image size as (width, height)
    
    Returns:
        Base64-encoded PNG string, or None if rendering fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        img = Draw.MolToImage(mol, size=size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES and return RDKit Mol, or None if invalid."""
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES is valid (parsable by RDKit)."""
    return smiles_to_mol(smiles) is not None