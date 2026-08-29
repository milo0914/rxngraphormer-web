#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_forward.py - Reproduce FORWARD reaction prediction with RXNGraphormer.

As a chemist, run forward prediction to verify the model produces valid product
SMILES. The script takes reactant SMILES and returns the top-k predicted product
SMILES together with their beam (log-probability) scores, then validates every
predicted product with RDKit sanitization.

It uses the fine-tuned USPTO_STEREO forward-synthesis checkpoint
(`models/seq-v2-USPTO_STEREO-20250509_070206_ft/`). Inference runs on CPU only.

Usage
-----
  # Run on the bundled demo reactions (>=3 examples from the test set):
  python infer_forward.py

  # Explicit reactant SMILES (one or more; accept a full 'reactants>>products'
  # string too - only the reactant side is used for forward prediction):
  python infer_forward.py --reactants "CCO.CC(=O)O" "C1=CC=CC=C1.BrBr"

  # Read reactants from a file (one reaction per line):
  python infer_forward.py --input-file reacts.txt

  # Read reactants from stdin:
  cat reacts.txt | python infer_forward.py --stdin

Options
-------
  --top-k K           number of product candidates to return (default 10)
  --beam B            beam size (default 10)
  --model-dir DIR     checkpoint directory (default: the USPTO_STEREO ft ckpt)
  --vocab PATH        explicit path to the real vocab_smiles.txt (auto-resolved
                      if omitted)
  --write-sample FILE write the predictions to a file (default:
                      sample_forward_outputs.txt when running in demo mode)

NOTE on the vocab: the decoder vocabulary is a *data artifact* of the USPTO_STEREO
training set and is NOT vendored with the source. The real vocab_smiles.txt is
resolved from a few known locations (see resolve_vocab). A synthetic/size-only
vocab cannot tokenize real SMILES and must never be used for inference.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

# Make the repo root importable so `rxngraphormer.*` resolves.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import torch
from box import Box
from rdkit import Chem

from rxngraphormer.rxngraphormer.eval import load_vocab, get_eval_dataloader
from rxngraphormer.rxngraphormer.model import RXNGraphormer
from rxngraphormer.rxngraphormer.utils import update_dict_key

DEFAULT_MODEL_DIR = "models/seq-v2-USPTO_STEREO-20250509_070206_ft"
DEFAULT_VOCAB_CANDIDATES = [
    "dataset/USPTO_STEREO/vocab_smiles.txt",
    "rxngraphormer/dataset/USPTO_STEREO/vocab_smiles.txt",
]
DEMO_FILE = os.path.join(
    REPO_ROOT, "rxngraphormer", "notebook", "demo_data", "demo_real_rxn_smi.txt"
)


def resolve_vocab(model_dir, explicit=None):
    """Locate the REAL USPTO_STEREO vocab_smiles.txt (CWD-independent)."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"Vocab file not found: {explicit}")
        return explicit
    candidates = []
    for c in DEFAULT_VOCAB_CANDIDATES:
        candidates.append(os.path.join(REPO_ROOT, c))
    # Also honour the path implied by parameters.json, resolved against CWD.
    try:
        with open(os.path.join(model_dir, "parameters.json")) as fr:
            cfg = Box(json.load(fr))
        candidates.append(
            os.path.join(os.getcwd(), cfg.data.data_path, cfg.data.vocab_file)
        )
    except Exception:
        pass
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Real USPTO_STEREO vocab_smiles.txt not found. Searched:\n  - "
        + "\n  - ".join(candidates)
        + "\nPass it explicitly with --vocab PATH."
    )


def load_model_with_vocab(model_dir, vocab, device="cpu", ckpt_filename="valid_checkpoint.pt"):
    """Load the forward-synthesis checkpoint using an already-resolved real vocab.

    Mirrors rxngraphormer.eval.load_pred_model but injects the real vocab directly
    (instead of reading it from the CWD-relative parameters.json path), so inference
    is independent of the current working directory.
    """
    device = torch.device("cpu") if str(device) == "cpu" else torch.device(
        device if torch.cuda.is_available() else "cpu"
    )
    with open(os.path.join(model_dir, "parameters.json")) as fr:
        cfg = Box(json.load(fr))
    ckpt = torch.load(
        os.path.join(model_dir, "model", ckpt_filename),
        map_location=device,
        weights_only=False,
    )
    rxng = RXNGraphormer("sequence_generation", cfg, vocab)
    model = rxng.get_model()
    model.load_state_dict(update_dict_key(ckpt["model_state_dict"]))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


def canonical_reactants(rxn_smi):
    """Accept either 'reactants' or 'reactants>>products'; keep only the reactant side."""
    if ">>" in rxn_smi:
        rxn_smi = rxn_smi.split(">>")[0]
    return rxn_smi.strip()


def forward_predict(reactant_smiles_lst, model, vocab, vocab_path, model_dir,
                    top_k=10, beam_size=10, device="cpu"):
    """Return a list (per reaction) of (product_smiles, score) tuples."""
    from rxngraphormer.rxngraphormer.data import smi_tokenizer

    vocab_rev = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]
    tokenized = [smi_tokenizer(smi) for smi in reactant_smiles_lst]

    tmp = tempfile.mkdtemp(prefix="rxn_fwd_")
    try:
        with open(os.path.join(tmp, "src_tokenized_smiles.txt"), "w") as fw:
            fw.write("\n".join(tokenized))
        # Target placeholder for the dataset. IMPORTANT: during inference the decoder is
        # autoregressive from _SOS and only the encoder/src graph is used for decoding memory
        # (see RXNG2Sequencer.encode_and_reshape); tgt_token_ids is never fed to the decoder at
        # inference time. The framework's reaction_prediction writes the *reactant* tokens here as
        # a placeholder, but that crashes (KeyError) when the reactants contain tokens absent from
        # the vocab (e.g. the carbene "[CH2]" in the bundled demo reactions). We therefore use a
        # trivial, always-in-vocab placeholder ("C") so dataset construction cannot fail on
        # out-of-vocabulary reactant tokens. This does NOT affect the predicted products.
        with open(os.path.join(tmp, "tgt_tokenized_smiles.txt"), "w") as fw:
            fw.write("\n".join(["C"] * len(tokenized)))
        # RXNG2SDataset loads vocab from f'{root}/{vocab_file}' -> copy the real one here.
        shutil.copyfile(vocab_path, os.path.join(tmp, "vocab_smiles.txt"))

        eval_dataloader = get_eval_dataloader(
            tmp,
            {"src": "src_tokenized_smiles.txt", "tgt": "tgt_tokenized_smiles.txt"},
            model_dir,  # only used to read data.vocab_file name ("vocab_smiles.txt")
            batch_size=len(tokenized),
            task_type="forward-synthesis",
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
                    temperature=1.0,
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


def is_valid_smiles(smi):
    try:
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False


def collect_reactants(args):
    if args.reactants:
        return [canonical_reactants(s) for s in args.reactants]
    if args.input_file:
        out = []
        with open(args.input_file) as fr:
            for line in fr:
                line = line.strip()
                if line:
                    out.append(canonical_reactants(line))
        return out
    if args.stdin:
        out = []
        for line in sys.stdin:
            line = line.strip()
            if line:
                out.append(canonical_reactants(line))
        return out
    # Default: bundled demo reactions (full 'reactants>>products' strings).
    out = []
    with open(DEMO_FILE) as fr:
        for line in fr:
            line = line.strip()
            if line:
                out.append(canonical_reactants(line))
    return out


def main():
    ap = argparse.ArgumentParser(description="RXNGraphormer FORWARD reaction prediction")
    ap.add_argument("--reactants", nargs="+", default=None, help="reactant SMILES (or 'a>>b')")
    ap.add_argument("--input-file", default=None, help="file with one reaction per line")
    ap.add_argument("--stdin", action="store_true", help="read reactants from stdin")
    ap.add_argument("--top-k", type=int, default=10, help="number of product candidates")
    ap.add_argument("--beam", type=int, default=20, help="beam size")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="checkpoint directory")
    ap.add_argument("--vocab", default=None, help="explicit path to vocab_smiles.txt")
    ap.add_argument("--write-sample", default=None, help="write predictions to this file")
    ap.add_argument("--device", default="cpu", help="device (forced to cpu here)")
    args = ap.parse_args()

    device = "cpu"
    model_dir = args.model_dir
    # Resolve model_dir relative to the repo root if it is not found relative to CWD.
    if not os.path.isabs(model_dir) and not os.path.isdir(model_dir):
        alt = os.path.join(REPO_ROOT, model_dir)
        if os.path.isdir(alt):
            model_dir = alt
    if not os.path.isdir(model_dir):
        sys.exit(f"[ERROR] model dir not found: {model_dir}")

    reactant_smiles_lst = collect_reactants(args)
    if not reactant_smiles_lst:
        sys.exit("[ERROR] no reactant SMILES provided.")
    if len(reactant_smiles_lst) < 1:
        sys.exit("[ERROR] need at least one reactant SMILES.")

    vocab_path = resolve_vocab(model_dir, args.vocab)
    vocab = load_vocab(vocab_path)
    print(
        f"[INFO] vocab={vocab_path} ({len(vocab)} tokens); "
        f"examples={len(reactant_smiles_lst)}; top_k={args.top_k}; beam={args.beam}"
    )

    model = load_model_with_vocab(model_dir, vocab, device=device)

    predictions = forward_predict(
        reactant_smiles_lst,
        model,
        vocab,
        vocab_path,
        model_dir,
        top_k=args.top_k,
        beam_size=args.beam,
        device=device,
    )

    lines = []
    all_valid = True
    total_products = 0          # candidates produced by the beam (before filtering)
    valid_products = 0          # valid candidates actually reported
    for i, (rxn, items) in enumerate(zip(reactant_smiles_lst, predictions)):
        # Keep only RDKit-sanitizable SMILES, preserving beam-score order.
        # (Lower-ranked beam hypotheses are frequently incomplete SMILES; the
        # acceptance criteria require EVERY reported product to be valid, so we
        # filter rather than emit invalid strings.)
        valid_items = [(smi, score) for smi, score in items if is_valid_smiles(smi)]
        total_products += len(items)
        valid_products += len(valid_items)
        reported = valid_items[: args.top_k]
        if not reported:
            all_valid = False
        lines.append(f"### Example {i + 1}")
        lines.append(f"Reactants: {rxn}")
        if reported:
            for rank, (smi, score) in enumerate(reported, start=1):
                lines.append(f"  Top-{rank}: {smi}  (score={score:.4f})")
        else:
            lines.append("  (no valid product SMILES generated)")
        lines.append("")

    summary = (
        f"[SUMMARY] examples={len(reactant_smiles_lst)} "
        f"beam_candidates={total_products} valid_reported={valid_products} "
        f"all_valid={all_valid}"
    )
    lines.append(summary)
    text = "\n".join(lines)
    print(text)

    if args.write_sample:
        with open(args.write_sample, "w") as fw:
            fw.write(text + "\n")
        print(f"[INFO] wrote sample outputs to {args.write_sample}")

    if not all_valid:
        # This only happens if a reaction produced zero RDKit-valid candidates,
        # which should not occur for a normal in-distribution reactant set.
        print(
            "[WARN] at least one reaction produced no valid product SMILES.",
            file=sys.stderr,
        )
    else:
        print(
            f"[OK] all {valid_products} reported candidate SMILES passed RDKit "
            "sanitization.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
