#!/usr/bin/env python3
"""Static structure check for the vendored RXNGraphormer framework (US-003).

Verifies, WITHOUT importing torch / OpenNMT-py / torch_geometric (which are
installed in later stories), that:

  1. the vendored tree exists at ``rxngraphormer/`` with the expected layout,
  2. every vendored ``*.py`` file parses with the local Python,
  3. the inference entry points documented in PROJECT_NOTES.md really exist
     (top-level functions / classes are located via AST, not via import),
  4. the eval config files for forward / retro sequence generation are present
     and are valid JSON.

Usage:
    python scripts/check_framework.py            # exit 0 == all checks pass
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "rxngraphormer"          # vendored repo root
PKG_ROOT = VENDOR_ROOT / "rxngraphormer"           # importable python package

# Files that must exist in the vendored tree.
REQUIRED_FILES = [
    VENDOR_ROOT / "LICENSE",
    VENDOR_ROOT / "README.md",
    VENDOR_ROOT / "setup.py",
    VENDOR_ROOT / "eval_model.py",
    VENDOR_ROOT / "train_model.py",
    VENDOR_ROOT / "data_preprocess.py",
    VENDOR_ROOT / "requirements_pt221.txt",
    PKG_ROOT / "__init__.py",
    PKG_ROOT / "model.py",
    PKG_ROOT / "eval.py",
    PKG_ROOT / "data.py",
    PKG_ROOT / "rxn_emb.py",
    PKG_ROOT / "utils.py",
    PKG_ROOT / "layer.py",
    PKG_ROOT / "train.py",
    PKG_ROOT / "midgen" / "midmol.py",
    PKG_ROOT / "midgen" / "collections" / "LRT_library.csv",
    PKG_ROOT / "midgen" / "collections" / "MT_library.csv",
]

# module (relative to the python package) -> required top-level symbols
REQUIRED_SYMBOLS = {
    "eval.py": [
        "reaction_prediction",   # one-shot forward / retro inference on SMILES
        "load_pred_model",       # checkpoint loader (US-005)
        "get_eval_dataloader",   # SMILES -> graph batches
        "SeqEval",               # dataset-level beam-search evaluation
        "eval_regression_performance",
    ],
    "model.py": [
        "RXNGraphormer",         # factory: task -> model
        "RXNG2Sequencer",         # graph2seq model (forward / retro)
        "RXNGRegressor",          # reactivity / selectivity model
    ],
    "data.py": [
        "smi_tokenizer",
        "load_vocab",
        "RXNG2SDataset",
    ],
    # NOTE: the pytorch2 branch dropped `RXNClassifier` (fictitious-reaction detection),
    # which only exists on the main/PyTorch-1 branch. It is not needed for the web app.
    "rxn_emb.py": ["RXNEMB"],
}

# Sequence-generation eval configs shipped upstream (forward + retro).
REQUIRED_CONFIGS = [
    "config/uspto_50k_eval.json",     # retro-synthesis (USPTO-50k)
    "config/uspto_full_eval.json",    # retro-synthesis (USPTO-full)
    "config/uspto_480k_eval.json",    # forward-synthesis (USPTO-480k)
    "config/uspto_stereo_eval.json",  # forward-synthesis (USPTO-STEREO)
]

failures: list[str] = []
checks = 0


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  [FAIL] {msg}")


def top_level_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def main() -> int:
    global checks
    print(f"RXNGraphormer vendored-source check (repo: {REPO_ROOT})")

    print("\n1) required files present")
    for path in REQUIRED_FILES:
        checks += 1
        rel = path.relative_to(REPO_ROOT)
        if path.is_file():
            ok(str(rel))
        else:
            fail(f"missing file: {rel}")

    print("\n2) no nested git repository (source is tracked by THIS repo)")
    checks += 1
    if (VENDOR_ROOT / ".git").exists():
        fail("rxngraphormer/.git still present - remove it so files are tracked here")
    else:
        ok("rxngraphormer/.git absent")

    print("\n3) every vendored python file parses")
    py_files = sorted(VENDOR_ROOT.rglob("*.py"))
    checks += 1
    if not py_files:
        fail("no python files found under rxngraphormer/")
    else:
        bad = []
        for path in py_files:
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:  # pragma: no cover - defensive
                bad.append(f"{path.relative_to(REPO_ROOT)}: {exc}")
        if bad:
            for item in bad:
                fail(f"syntax error: {item}")
        else:
            ok(f"{len(py_files)} python files parsed cleanly")

    print("\n4) documented entry points exist")
    for module, symbols in REQUIRED_SYMBOLS.items():
        path = PKG_ROOT / module
        if not path.is_file():
            checks += 1
            fail(f"cannot inspect missing module {module}")
            continue
        found = top_level_symbols(path)
        for symbol in symbols:
            checks += 1
            if symbol in found:
                ok(f"rxngraphormer/{module}::{symbol}")
            else:
                fail(f"symbol not found: rxngraphormer/{module}::{symbol}")

    print("\n5) sequence-generation eval configs are valid JSON")
    for rel in REQUIRED_CONFIGS:
        checks += 1
        path = VENDOR_ROOT / rel
        if not path.is_file():
            fail(f"missing config: {rel}")
            continue
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {rel}: {exc}")
            continue
        if cfg.get("task") != "sequence_generation":
            fail(f"{rel}: expected task == 'sequence_generation', got {cfg.get('task')!r}")
        else:
            ok(f"{rel} (model_path={cfg.get('trained_model_path')})")

    print("\n6) vendor provenance recorded")
    for rel in ("PROJECT_NOTES.md", "rxngraphormer/VENDOR_INFO.txt"):
        checks += 1
        path = REPO_ROOT / rel
        if not path.is_file():
            fail(f"missing {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if "ab6d13d1900a2a3125df24dc798ffa8a6cc6761d" in text and "licheng-xu-echo/RXNGraphormer" in text:
            ok(f"{rel} records repo URL + commit")
        else:
            fail(f"{rel} does not record the vendored repo URL and commit hash")

    print("\n" + "-" * 60)
    if failures:
        print(f"FAILED: {len(failures)} of {checks} checks failed")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"PASSED: all {checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
