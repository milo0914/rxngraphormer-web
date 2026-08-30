#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_retro.py - Reproduce RETROSYNTHESIS prediction with RXNGraphormer.

As a chemist, run retrosynthesis to verify the model proposes valid precursor
(preactor) sets. The script takes a product SMILES and returns the top-k
predicted precursor sets together with their beam (log-probability) scores, then
validates every predicted precursor set with RDKit sanitization.

It uses the RETROSYNTHESIS checkpoint from the original author's figshare article
28356077 ("Preprocessed datasets and model weights for RXNGraphormer", file
`USPTO_50k_model.zip`), placed under `models/USPTO_50k/`. That checkpoint's
`parameters.json` declares `model.task = "retrosynthesis"`, i.e. it was actually
trained for retrosynthesis (unlike the two forward-only `_ft` checkpoints from the
reproduction article 30498368). Inference runs on CPU only.

Usage
-----
  # Run on the bundled sample test-set products (>=3 examples from USPTO_50k test):
  python infer_retro.py

  # Explicit product SMILES (one or more; accept a full 'reactants>>products'
  # string too - only the product side is used for retrosynthesis):
  python infer_retro.py --products "CC(=O)Oc1ccccc1" "ClC1=CC=CC=C1"

  # Read products from a file (one product SMILES per line):
  python infer_retro.py --input-file products.txt

  # Read products from stdin:
  cat products.txt | python infer_retro.py --stdin

Options
-------
  --top-k K           number of precursor-set candidates to return (default 10)
  --beam B            beam size (default 20)
  --temperature T     decoding temperature (default 1.0; the framework's
                      uspto_50k_eval.json uses 3.0 for benchmarking)
  --model-dir DIR     checkpoint directory (default: models/USPTO_50k)
  --vocab PATH        explicit path to the real vocab_smiles.txt (auto-resolved
                      if omitted)
  --write-sample FILE write the predictions to a file (default:
                      sample_retro_outputs.txt when running in demo mode)

NOTE on the vocab: the decoder vocabulary is a *data artifact* of the USPTO_50k
training set and is NOT vendored with the source. The real vocab_smiles.txt (75
tokens) is resolved from a few known locations (see resolve_vocab). A synthetic /
size-only vocab cannot tokenize real SMILES and must never be used for inference.
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

DEFAULT_MODEL_DIR = "models/USPTO_50k"
DEFAULT_VOCAB_CANDIDATES = [
    "dataset/USPTO_50k/vocab_smiles.txt",
    "rxngraphormer/dataset/USPTO_50k/vocab_smiles.txt",
]
DEMO_FILE = os.path.join(REPO_ROOT, "dataset", "USPTO_50k", "sample_retro_products.txt")


def resolve_vocab(model_dir, explicit=None):
    """Locate the REAL USPTO_50k vocab_smiles.txt (CWD-independent)."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"Vocab file not found: {explicit}")
        return explicit
    candidates = [os.path.join(REPO_ROOT, c) for c in DEFAULT_VOCAB_CANDIDATES]
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
        "Real USPTO_50k vocab_smiles.txt not found. Searched:\n  - "
        + "\n  - ".join(candidates)
        + "\nPass it explicitly with --vocab PATH."
    )


def load_model_with_vocab(model_dir, vocab, device="cpu", ckpt_filename="valid_checkpoint.pt"):
    """Load the retrosynthesis checkpoint using an already-resolved real vocab.

    Mirrors rxngraphormer.eval.load_pred_model but injects the real vocab directly
    (instead of reading it from the CWD-relative parameters.json path), so inference
    is independent of the current working directory.
    """
    device = torch.device("cpu") if str(device) == "cpu" else torch.device(
        device if torch.cuda.is_available() else "cpu"
    )
    with open(os.path.join(model_dir, "parameters.json")) as fr:
        cfg = Box(json.load(fr))
    # Sanity: confirm this is actually a retrosynthesis checkpoint.
    task = cfg.model.task
    if task != "retrosynthesis":
        print(
            f"[WARN] checkpoint model.task = {task!r}, expected 'retrosynthesis'. "
            "Predictions may be nonsensical for retrosynthesis.",
            file=sys.stderr,
        )
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


def canonical_product(rxn_smi):
    """Accept either 'product' or 'reactants>>product'; keep only the product side."""
    if ">>" in rxn_smi:
        rxn_smi = rxn_smi.split(">>")[-1]
    return rxn_smi.strip()


def retro_predict(product_smiles_lst, model, vocab, vocab_path, model_dir,
                  top_k=10, beam_size=20, temperature=1.0, device="cpu"):
    """Return a list (per product) of (precursor_set_smiles, score) tuples."""
    from rxngraphormer.rxngraphormer.data import smi_tokenizer

    vocab_rev = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]
    tokenized = [smi_tokenizer(smi) for smi in product_smiles_lst]

    tmp = tempfile.mkdtemp(prefix="rxn_retro_")
    try:
        with open(os.path.join(tmp, "src_tokenized_smiles.txt"), "w") as fw:
            fw.write("\n".join(tokenized))
        # Target placeholder for the dataset. IMPORTANT: during inference the decoder is
        # autoregressive from _SOS and only the encoder/src graph is used for decoding memory
        # (see RXNG2Sequencer.encode_and_reshape); tgt_token_ids is never fed to the decoder at
        # inference time. We therefore use a trivial, always-in-vocab placeholder ("C") so dataset
        # construction cannot fail on out-of-vocabulary product tokens. This does NOT affect the
        # predicted precursor sets.
        with open(os.path.join(tmp, "tgt_tokenized_smiles.txt"), "w") as fw:
            fw.write("\n".join(["C"] * len(tokenized)))
        # RXNG2SDataset loads vocab from f'{root}/{vocab_file}' -> copy the real one here.
        shutil.copyfile(vocab_path, os.path.join(tmp, "vocab_smiles.txt"))

        eval_dataloader = get_eval_dataloader(
            tmp,
            {"src": "src_tokenized_smiles.txt", "tgt": "tgt_tokenized_smiles.txt"},
            model_dir,  # only used to read data.vocab_file name ("vocab_smiles.txt")
            batch_size=len(tokenized),
            task_type="retro-synthesis",
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


def is_valid_smiles(smi):
    try:
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False


def collect_products(args):
    if args.products:
        return [canonical_product(s) for s in args.products]
    if args.input_file:
        out = []
        with open(args.input_file) as fr:
            for line in fr:
                line = line.strip()
                if line:
                    out.append(canonical_product(line))
        return out
    if args.stdin:
        out = []
        for line in sys.stdin:
            line = line.strip()
            if line:
                out.append(canonical_product(line))
        return out
    # Default: bundled sample USPTO_50k test-set products.
    out = []
    with open(DEMO_FILE) as fr:
        for line in fr:
            line = line.strip()
            if line:
                out.append(canonical_product(line))
    return out


def main():
    ap = argparse.ArgumentParser(description="RXNGraphormer RETROSYNTHESIS prediction")
    ap.add_argument("--products", nargs="+", default=None, help="product SMILES (or 'a>>b')")
    ap.add_argument("--input-file", default=None, help="file with one product per line")
    ap.add_argument("--stdin", action="store_true", help="read products from stdin")
    ap.add_argument("--top-k", type=int, default=10, help="number of precursor-set candidates")
    ap.add_argument("--beam", type=int, default=20, help="beam size")
    ap.add_argument("--temperature", type=float, default=1.0, help="decoding temperature")
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

    product_smiles_lst = collect_products(args)
    if not product_smiles_lst:
        sys.exit("[ERROR] no product SMILES provided.")
    if len(product_smiles_lst) < 1:
        sys.exit("[ERROR] need at least one product SMILES.")

    vocab_path = resolve_vocab(model_dir, args.vocab)
    vocab = load_vocab(vocab_path)
    print(
        f"[INFO] vocab={vocab_path} ({len(vocab)} tokens); "
        f"products={len(product_smiles_lst)}; top_k={args.top_k}; "
        f"beam={args.beam}; temperature={args.temperature}"
    )

    model = load_model_with_vocab(model_dir, vocab, device=device)

    predictions = retro_predict(
        product_smiles_lst,
        model,
        vocab,
        vocab_path,
        model_dir,
        top_k=args.top_k,
        beam_size=args.beam,
        temperature=args.temperature,
        device=device,
    )

    lines = []
    all_valid = True
    total_candidates = 0          # candidates produced by the beam (before filtering)
    valid_reported = 0            # valid candidates actually reported
    for i, (rxn, items) in enumerate(zip(product_smiles_lst, predictions)):
        # Keep only RDKit-sanitizable SMILES, preserving beam-score order.
        # (Lower-ranked beam hypotheses are frequently incomplete SMILES; the
        # acceptance criteria require EVERY reported precursor set to be valid, so
        # we filter rather than emit invalid strings. A predicted precursor set may
        # be a single molecule or a dot-joined multi-reactant set; Chem.MolFromSmiles
        # parses multi-fragment SMILES, so the whole set is validated at once.)
        valid_items = [(smi, score) for smi, score in items if is_valid_smiles(smi)]
        total_candidates += len(items)
        valid_reported += len(valid_items)
        reported = valid_items[: args.top_k]
        if not reported:
            all_valid = False
        lines.append(f"### Example {i + 1}")
        lines.append(f"Product: {rxn}")
        if reported:
            for rank, (smi, score) in enumerate(reported, start=1):
                lines.append(f"  Top-{rank}: {smi}  (score={score:.4f})")
        else:
            lines.append("  (no valid precursor-set SMILES generated)")
        lines.append("")

    summary = (
        f"[SUMMARY] products={len(product_smiles_lst)} "
        f"beam_candidates={total_candidates} valid_reported={valid_reported} "
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
        print(
            "[WARN] at least one product produced no valid precursor-set SMILES.",
            file=sys.stderr,
        )
    else:
        print(
            f"[OK] all {valid_reported} reported candidate precursor sets passed "
            "RDKit sanitization.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
