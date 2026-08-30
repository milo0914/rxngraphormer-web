"""Model loading utilities for RXNGraphormer checkpoints."""
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from box import Box
from rdkit import Chem

from rxngraphormer.rxngraphormer.data import load_vocab, smi_tokenizer
from rxngraphormer.rxngraphormer.eval import get_eval_dataloader
from rxngraphormer.rxngraphormer.model import RXNGraphormer
from rxngraphormer.rxngraphormer.utils import update_dict_key


# ─── Globals (set at startup) ─────────────────────────────────────────────

_forward_model = None
_forward_vocab = None
_forward_vocab_rev = None
_forward_model_dir = None
_forward_vocab_path = None

_retro_model = None
_retro_vocab = None
_retro_vocab_rev = None
_retro_model_dir = None
_retro_vocab_path = None


# ─── Helpers ──────────────────────────────────────────────────────────────

def _resolve_model_dir(model_dir: str) -> Path:
    """Resolve model directory relative to repo root if not found."""
    repo_root = Path(__file__).parent.parent.parent
    path = Path(model_dir)
    if path.is_dir():
        return path
    alt = repo_root / model_dir
    if alt.is_dir():
        return alt
    raise FileNotFoundError(f"Model directory not found: {model_dir} (tried {path} and {alt})")


def _resolve_vocab_path(vocab_path: str, model_dir: Path) -> Path:
    """Resolve vocab path relative to repo root."""
    repo_root = Path(__file__).parent.parent.parent
    path = Path(vocab_path)
    if path.is_file():
        return path
    # Try relative to repo root
    alt = repo_root / vocab_path
    if alt.is_file():
        return alt
    # Try relative to model_dir from parameters.json
    try:
        with open(model_dir / "parameters.json") as f:
            cfg = Box(json.load(f))
        implied = Path(cfg.data.data_path) / cfg.data.vocab_file
        alt2 = Path.cwd() / implied
        if alt2.is_file():
            return alt2
        alt3 = repo_root / implied
        if alt3.is_file():
            return alt3
    except Exception:
        pass
    raise FileNotFoundError(f"Vocab file not found: {vocab_path}")


def _load_vocab_and_rev(vocab_path: Path) -> Tuple[Dict[str, int], List[str]]:
    """Load vocab dict and reverse lookup list."""
    vocab = load_vocab(str(vocab_path))
    vocab_rev = [k for k, v in sorted(vocab.items(), key=lambda t: t[1])]
    return vocab, vocab_rev


def _load_model(
    model_dir: Path,
    vocab: Dict[str, int],
    ckpt_filename: str = "valid_checkpoint.pt",
    device: str = "cpu",
) -> torch.nn.Module:
    """Load RXNGraphormer model from checkpoint."""
    device = torch.device("cpu") if str(device) == "cpu" else torch.device(device)
    
    with open(model_dir / "parameters.json") as f:
        cfg = Box(json.load(f))
    
    ckpt = torch.load(
        model_dir / "model" / ckpt_filename,
        map_location=device,
        weights_only=False,
    )
    
    rxng = RXNGraphormer("sequence_generation", cfg, vocab)
    model = rxng.get_model()
    model.load_state_dict(update_dict_key(ckpt["model_state_dict"]))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_dir}!")
    return model


# ─── Public API ───────────────────────────────────────────────────────────

def load_forward_model(
    model_dir: str,
    vocab_path: str,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
    """
    Load the forward-synthesis model and vocab.
    
    Returns:
        (model, vocab, vocab_rev)
    """
    global _forward_model, _forward_vocab, _forward_vocab_rev
    global _forward_model_dir, _forward_vocab_path
    
    _forward_model_dir = _resolve_model_dir(model_dir)
    _forward_vocab_path = _resolve_vocab_path(vocab_path, _forward_model_dir)
    
    _forward_vocab, _forward_vocab_rev = _load_vocab_and_rev(_forward_vocab_path)
    _forward_model = _load_model(_forward_model_dir, _forward_vocab, device=device)
    
    return _forward_model, _forward_vocab, _forward_vocab_rev


def load_retro_model(
    model_dir: str,
    vocab_path: str,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
    """
    Load the retrosynthesis model and vocab.
    
    Returns:
        (model, vocab, vocab_rev)
    """
    global _retro_model, _retro_vocab, _retro_vocab_rev
    global _retro_model_dir, _retro_vocab_path
    
    _retro_model_dir = _resolve_model_dir(model_dir)
    _retro_vocab_path = _resolve_vocab_path(vocab_path, _retro_model_dir)
    
    # Verify it's a retrosynthesis checkpoint
    with open(_retro_model_dir / "parameters.json") as f:
        cfg = Box(json.load(f))
    task = cfg.model.task
    if task != "retrosynthesis":
        print(f"[WARN] checkpoint model.task = {task!r}, expected 'retrosynthesis'")
    
    _retro_vocab, _retro_vocab_rev = _load_vocab_and_rev(_retro_vocab_path)
    _retro_model = _load_model(_retro_model_dir, _retro_vocab, device=device)
    
    return _retro_model, _retro_vocab, _retro_vocab_rev


def get_forward_model() -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
    """Get loaded forward model, vocab, and vocab_rev."""
    if _forward_model is None:
        raise RuntimeError("Forward model not loaded")
    return _forward_model, _forward_vocab, _forward_vocab_rev


def get_retro_model() -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
    """Get loaded retro model, vocab, and vocab_rev."""
    if _retro_model is None:
        raise RuntimeError("Retro model not loaded")
    return _retro_model, _retro_vocab, _retro_vocab_rev


def is_forward_loaded() -> bool:
    return _forward_model is not None


def is_retro_loaded() -> bool:
    return _retro_model is not None


# ─── Inference Core (shared) ──────────────────────────────────────────────

def _run_inference(
    smiles_list: List[str],
    model: torch.nn.Module,
    vocab_rev: List[str],
    vocab_path: Path,
    model_dir: Path,
    task_type: str,  # "forward-synthesis" or "retro-synthesis"
    top_k: int,
    beam_size: int,
    temperature: float = 1.0,
    device: str = "cpu",
) -> List[List[Tuple[str, float]]]:
    """
    Run inference on a list of SMILES.
    
    Returns:
        List of (per-SMILES) list of (smiles, score) tuples, score-ordered.
    """
    device = torch.device("cpu") if str(device) == "cpu" else torch.device(device)
    
    # Tokenize input SMILES
    tokenized = [smi_tokenizer(smi) for smi in smiles_list]
    
    # Create temporary dataset directory
    tmp = tempfile.mkdtemp(prefix="rxn_api_")
    try:
        # Write source tokenized SMILES
        with open(Path(tmp) / "src_tokenized_smiles.txt", "w") as fw:
            fw.write("\n".join(tokenized))
        
        # Write dummy target (always "C" - in-vocab placeholder)
        # The decoder is autoregressive from _SOS; tgt is only used for dataset construction,
        # never fed to decoder during inference.
        with open(Path(tmp) / "tgt_tokenized_smiles.txt", "w") as fw:
            fw.write("\n".join(["C"] * len(tokenized)))
        
        # Copy vocab to temp dir (RXNG2SDataset loads from f'{root}/{vocab_file}')
        shutil.copyfile(vocab_path, Path(tmp) / "vocab_smiles.txt")
        
        # Create dataloader
        eval_dataloader = get_eval_dataloader(
            tmp,
            {"src": "src_tokenized_smiles.txt", "tgt": "tgt_tokenized_smiles.txt"},
            str(model_dir),
            batch_size=len(tokenized),
            task_type=task_type,
        )
        
        all_predictions = []
        with torch.no_grad():
            for batch_data in eval_dataloader:
                batch_data = batch_data.to(device)
                results = model.infer(
                    reaction_batch=batch_data,
                    batch_size=len(batch_data.tgt_lens),
                    beam_size=beam_size,
                    n_best=top_k,
                    temperature=temperature,
                    min_length=1,
                    max_length=512,
                )
                for beam_preds, beam_scores in zip(
                    results["predictions"], results["scores"]
                ):
                    items = []
                    for pred, sc in zip(beam_preds, beam_scores):
                        idx = pred.detach().cpu().numpy()
                        toks = [vocab_rev[int(i)] for i in idx[:-1]]  # drop _EOS
                        smi = "".join(toks)
                        items.append((smi, float(sc)))
                    all_predictions.append(items)
        return all_predictions
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── Public Inference Functions ───────────────────────────────────────────

def forward_predict(
    reactants: str,
    top_k: int = 5,
    beam_size: int = 20,
    temperature: float = 1.0,
    device: str = "cpu",
) -> List[Tuple[str, float]]:
    """
    Run forward prediction on reactant SMILES.
    
    Args:
        reactants: Reactant SMILES (dot-separated)
        top_k: Number of predictions to return
        beam_size: Beam width
        temperature: Decoding temperature
        device: Device to run on
    
    Returns:
        List of (product_smiles, score) tuples, score-ordered (highest first)
    """
    model, vocab, vocab_rev = get_forward_model()
    
    # Canonicalize: keep only reactant side if ">>" present
    if ">>" in reactants:
        reactants = reactants.split(">>")[0].strip()
    
    predictions = _run_inference(
        [reactants],
        model,
        vocab_rev,
        _forward_vocab_path,
        _forward_model_dir,
        "forward-synthesis",
        top_k=top_k,
        beam_size=beam_size,
        temperature=temperature,
        device=device,
    )
    return predictions[0]


def retro_predict(
    product: str,
    top_k: int = 5,
    beam_size: int = 20,
    temperature: float = 1.0,
    device: str = "cpu",
) -> List[Tuple[str, float]]:
    """
    Run retrosynthesis prediction on product SMILES.
    
    Args:
        product: Product SMILES
        top_k: Number of predictions to return
        beam_size: Beam width
        temperature: Decoding temperature
        device: Device to run on
    
    Returns:
        List of (precursor_set_smiles, score) tuples, score-ordered (highest first)
    """
    model, vocab, vocab_rev = get_retro_model()
    
    # Canonicalize: keep only product side if ">>" present
    if ">>" in product:
        product = product.split(">>")[-1].strip()
    
    predictions = _run_inference(
        [product],
        model,
        vocab_rev,
        _retro_vocab_path,
        _retro_model_dir,
        "retro-synthesis",
        top_k=top_k,
        beam_size=beam_size,
        temperature=temperature,
        device=device,
    )
    return predictions[0]