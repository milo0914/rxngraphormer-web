#!/usr/bin/env python
"""
load_checkpoint.py  (US-005)
-------------------------------
Load a fine-tuned RXNGraphormer forward-prediction checkpoint on CPU and print a
model summary + parameter count.

Design notes
============
* CPU only: every tensor/device is forced to ``cpu`` (the framework defaults to
  ``cuda:0``). We call the framework's own loader
  ``rxngraphormer.rxngraphormer.eval.load_pred_model`` so the load path exactly
  matches inference (US-006/US-010).
* The framework's loader reads the vocab from
  ``parameters.json -> data.data_path + data.vocab_file`` (relative to CWD).
  That file (``vocab_smiles.txt``) is a *data artifact* of the USPTO_STEREO
  training set and is NOT vendored with the source code, so it is absent here.
  For the purpose of LOADING the weights, only the vocabulary SIZE matters
  (it sizes the decoder embedding / output layer). We therefore derive the
  correct size from the checkpoint itself and generate a synthetic vocab of that
  exact size in a temp dir, then patch a copy of ``parameters.json`` to point at
  it. No fake vocab is ever written into the repository tree.
  -> US-006 (inference) must obtain the REAL ``vocab_smiles.txt`` from the
     USPTO_STEREO dataset for correct tokenisation.

Run:
    /app/venv/bin/python load_checkpoint.py
Exit code 0 on success.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Make the vendored framework importable.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "rxngraphormer"))

import torch  # noqa: E402

CKPT_DIR = REPO_ROOT / "models" / "seq-v2-USPTO_STEREO-20250509_070206_ft"
CKPT_FILE = "valid_checkpoint.pt"          # model/valid_checkpoint.pt
TASK_TYPE = "forward-synthesis"             # forward_prediction checkpoint
DEVICE = "cpu"


def derive_vocab_size(ckpt_dir: Path, ckpt_file: str) -> int:
    """Get the vocab size from the decoder embedding / output layer shape."""
    sd = torch.load(
        ckpt_dir / "model" / ckpt_file,
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]
    for key in (
        "decoder_embeddings.make_embedding.emb_luts.0.weight",
        "decoder.embeddings.make_embedding.emb_luts.0.weight",
        "output_layer.weight",
    ):
        if key in sd:
            return int(sd[key].shape[0])
    # Fallback: scan for any embedding lookup table.
    for key, val in sd.items():
        if key.endswith("emb_luts.0.weight") and val.dim() == 2:
            return int(val.shape[0])
    raise RuntimeError("Could not determine vocab size from checkpoint.")


def main() -> None:
    assert CKPT_DIR.exists(), f"Checkpoint dir not found: {CKPT_DIR}"

    vocab_size = derive_vocab_size(CKPT_DIR, CKPT_FILE)
    print(f"Derived vocab size from checkpoint: {vocab_size}")

    # Build a synthetic vocab of the exact required size. Only the special
    # tokens referenced by the model constructor (_PAD, _SOS, _EOS) matter for
    # loading; the filler rows are never indexed during a pure load.
    special = ["_PAD", "_SOS", "_EOS"]
    tokens = list(special) + [f"_T{i}" for i in range(len(special), vocab_size)]
    assert len(tokens) == vocab_size, "vocab size mismatch"

    # Prepare an isolated temp workspace so nothing is written into the repo.
    tmp = Path(tempfile.mkdtemp(prefix="rxng_load_"))
    vocab_path = tmp / "vocab_smiles.txt"
    vocab_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    # Copy + patch parameters.json so the loader finds our synthetic vocab.
    params = json.loads((CKPT_DIR / "parameters.json").read_text())
    params["data"]["data_path"] = str(tmp)
    params["data"]["vocab_file"] = "vocab_smiles.txt"
    (tmp / "parameters.json").write_text(json.dumps(params))

    # The loader reads model weights from <ckpt_dir>/model/<file>; expose the
    # real (large, gitignored) weights via a symlink instead of copying them.
    (tmp / "model").symlink_to(CKPT_DIR / "model", target_is_directory=True)

    # Import the framework's own loader (after sys.path is set up).
    from rxngraphormer.eval import load_pred_model  # noqa: E402

    model = load_pred_model(
        pretrained_model_path=str(tmp),
        ckpt_filename=CKPT_FILE,
        task_type=TASK_TYPE,
        device=DEVICE,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n=== Model summary ===")
    print(f"Task type           : {TASK_TYPE}")
    print(f"Device              : {DEVICE}")
    print(f"Total params        : {total:,}")
    print(f"Trainable params    : {trainable:,}")
    print("\n--- top-level modules ---")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:24s}: {n:,}")

    print("\nLOADED OK; total params =", total)


if __name__ == "__main__":
    main()
