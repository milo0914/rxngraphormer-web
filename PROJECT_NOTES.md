# PROJECT NOTES — RXNGraphormer Web

Engineering notes for reproducing RXNGraphormer (forward + retrosynthesis) and serving it
through a small web app. Updated per user story; newest sections appended.

---

## US-003 — Vendored RXNGraphormer framework source

### Chosen upstream repository

| Item | Value |
| --- | --- |
| Repo URL | <https://github.com/licheng-xu-echo/RXNGraphormer> |
| Why authoritative | "Official implementation" of the paper, by the first author (`licheng-xu-echo`); 56★, MIT licence, linked from the Nature Machine Intelligence article |
| Paper | Xu, LC., Tang, MJ., An, J. et al. *A unified pre-trained deep learning framework for cross-task reaction performance prediction and synthesis planning*, **Nat. Mach. Intell. 2025, 7, 1561** — <https://www.nature.com/articles/s42256-025-01098-4> |
| Vendored branch | `pytorch2` |
| **Vendored commit** | **`ab6d13d1900a2a3125df24dc798ffa8a6cc6761d`** (2026-08-24, "update: save meta results for sequence generation") |
| Vendored path | `rxngraphormer/` (inside this repo, nested `.git` removed → tracked by us) |
| Licence | MIT — `rxngraphormer/LICENSE` |
| Provenance file | `rxngraphormer/VENDOR_INFO.txt` |

Reference (not vendored): `main @ 0387656ea219b5526952ebd9828c9dba5bf8d689` — the PyTorch-1.12/CUDA-11.3
branch used in the paper. Upstream's own README says *"PyTorch 2 version ([pytorch2 branch]) is
recommended, which continues to be actively developed"*, and this box is **CPU-only with torch 2.x**,
so the `pytorch2` branch was vendored. `rxngraphormer/model.py` and `rxngraphormer/layer.py` are
**byte-identical** on both commits (verified by md5), i.e. published checkpoints remain loadable.

Related repos found during the search (for context only, NOT vendored):

* <https://github.com/TheLiaoGroup/RXNGraphormer-Reproduction> (mirror: `MJ-Zeng/RXNGraphormer-Reproduction`)
  — the *reusability report* reproduction (<https://www.nature.com/articles/s42256-026-01257-1>). It is a
  copy of the original code plus a `reproduction/` folder of notebooks/scripts. **Its Figshare deposit
  (article `30498368`, DOI `10.6084/m9.figshare.30498368`) is where US-004 fetches the checkpoints**, so
  the checkpoints we will use come from this reproduction of the framework we vendored.

### Vendored layout (importable structure)

```
rxngraphormer/                     <- vendored upstream repo root (run scripts from HERE)
├── VENDOR_INFO.txt                <- provenance (added by us, not upstream)
├── eval_model.py                  <- CLI: batch evaluation / inference (config-driven)
├── train_model.py, data_preprocess.py
├── config/*.json                  <- ready-made train/eval configs
├── dataset/  model_path/          <- EMPTY upstream placeholders; datasets + checkpoints go here
├── requirements_pt221.txt         <- torch 2.2.1 + PyG (CUDA 12.1 wheels; CPU equivalents needed here)
├── setup.py                       <- `pip install .` exposes package name `rxngraphormer` (v1.0.1)
└── rxngraphormer/                 <- the actual PYTHON PACKAGE (import rxngraphormer)
    ├── model.py                   <- RXNGraphormer (factory), RXNG2Sequencer, RXNGRegressor, .infer()
    ├── eval.py                    <- reaction_prediction(), load_pred_model(), get_eval_dataloader(), SeqEval
    ├── data.py                    <- smi_tokenizer(), load_vocab(), RXNG2SDataset, RXNDataset, collate fns
    ├── layer.py, train.py, scheduler.py, utils.py, ext_feat.py, rxn_emb.py (RXNEMB / RXNClassifier)
    └── midgen/                    <- "delta-mol"/mechanism SMILES generation (regression tasks only)
```

Import options (decide in US-005): either `pip install -e rxngraphormer/` (uses upstream `setup.py`;
its `install_requires` pins CUDA wheels, so prefer `pip install -e rxngraphormer/ --no-deps` after
installing CPU deps ourselves), or add `rxngraphormer/` to `sys.path` / `PYTHONPATH`.
The package directory is `rxngraphormer/rxngraphormer/`, i.e. `import rxngraphormer` works only when
the **vendored repo root** is on the path — mind the doubled directory name.

### Inference entry points

Both synthesis-planning directions use the SAME graph2seq model class and the SAME code path; only the
checkpoint and the `task_type` string differ (`"forward-synthesis"` vs `"retro-synthesis"`).

#### 1. One-shot SMILES API (what the web backend will wrap) — `rxngraphormer/rxngraphormer/eval.py:319`

```python
from rxngraphormer.eval import reaction_prediction

# FORWARD prediction: input = reactant SMILES (dot-separated), output = product candidates
fwd = reaction_prediction(
    "./model_path/USPTO_480k",                 # dir with parameters.json + model/valid_checkpoint.pt
    ["C1CCOC1.CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1.[Cl-]",
     "CN.O.O=C(O)c1ccc(Cl)c([N+](=O)[O-])c1"],  # >= 2 entries required (assert in code)
    task_type="forward-synthesis",
    device="cpu",                               # default is 'cuda:0' -> MUST pass 'cpu' here
    params={"batch_size": 4, "beam_size": 10, "n_best": 10,
            "temperature": 2.5, "min_length": 1, "max_length": 512},
)

# RETROSYNTHESIS: input = product SMILES, output = precursor-set candidates
retro = reaction_prediction(
    "./model_path/USPTO_50k",
    ["COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O",
     "O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"],
    task_type="retro-synthesis",
    device="cpu",
    params={"batch_size": 4, "beam_size": 10, "n_best": 10,
            "temperature": 3.0, "min_length": 1, "max_length": 512},
)
```

Returns a `pandas.DataFrame`: rows `Top-1 … Top-n_best`, one column per input SMILES, cells = predicted
SMILES strings (un-canonicalised; sanitise with RDKit ourselves). Upstream README timing: ~40 s / 100
reactions on a laptop CPU, so CPU-only serving is feasible.
**Note:** this function returns candidate SMILES but *not* beam scores — for the "ranked candidates +
scores" API contract (US-009/US-010) use the lower-level path below and read `results["scores"]` from
`model.infer(...)`.

#### 2. Lower-level path — recommended for the backend (avoids the `midgen` import)

```python
from rxngraphormer.eval import load_pred_model, get_eval_dataloader   # eval.py:237 / eval.py:295
model = load_pred_model("./model_path/USPTO_50k", ckpt_filename="valid_checkpoint.pt",
                        task_type="retro-synthesis", device="cpu")
loader = get_eval_dataloader("./tmp_dir", {"src": "src.txt", "tgt": "tgt.txt"},
                             pretrained_model_path="./model_path/USPTO_50k",
                             batch_size=4, task_type="retro-synthesis")
out = model.infer(reaction_batch=batch, batch_size=n, beam_size=10, n_best=10,
                  temperature=3.0, min_length=1, max_length=512)   # model.py:1136 (RXNG2Sequencer.infer)
# out["predictions"] -> token-id tensors, decode with vocab_rev; out["scores"] -> beam log-probs
```

Tokenisation/vocab helpers: `rxngraphormer.data.smi_tokenizer`, `rxngraphormer.data.load_vocab`;
model construction: `RXNGraphormer("sequence_generation", config, vocab).get_model()` (`model.py:1264`).

#### 3. Dataset-level CLI (accuracy reproduction, US-008) — `rxngraphormer/eval_model.py`

```bash
cd rxngraphormer/
python eval_model.py --config_json ./config/uspto_480k_eval.json   # FORWARD  (USPTO-480k)
python eval_model.py --config_json ./config/uspto_stereo_eval.json # FORWARD  (USPTO-STEREO, stereo-aware)
python eval_model.py --config_json ./config/uspto_50k_eval.json    # RETRO    (USPTO-50k)
python eval_model.py --config_json ./config/uspto_full_eval.json   # RETRO    (USPTO-full)
python eval_model.py --config_json ./config/buchwald_hartwig_eval.json  # regression (not needed for the web app)
```

`eval_model.py` builds `SeqEval` (`eval.py:13`) which loads the *whole preprocessed test set* declared in
the checkpoint's `parameters.json` and prints Top-1…Top-k accuracy (`save_prediction` dumps a CSV).
Config knobs used by the paper: `beam_size=30, n_best=30, topk=10`, `temperature` 2.5 (480k) / 2.75
(STEREO) / 3.0 (50k) / 1.0 (full).

Other entry points (not needed for the web UI): `reaction_prediction(..., task_type="reactivity" |
"selectivity")` for yield/selectivity regression, and `rxngraphormer.rxn_emb.RXNEMB` for reaction
embeddings. Branch difference to remember: `rxn_emb.RXNClassifier` (fictitious-reaction detection, shown
in the `main`-branch README) does **not** exist on the vendored `pytorch2` branch — that section was
dropped from its README too. Use the `main` branch if that feature is ever needed.

### Checkpoint directory contract (needed by US-004/US-005)

`load_pred_model` / `SeqEval` expect, for a checkpoint directory `M`:

```
M/parameters.json           # full training config (Box-loaded); contains data.data_path + data.vocab_file
M/model/valid_checkpoint.pt # torch.load(..., weights_only=False) -> dict with "model_state_dict"
```

**GOTCHA (biggest one):** the vocabulary is read from `parameters.json → data.data_path + data.vocab_file`,
e.g. `./dataset/USPTO_50k/vocab_smiles.txt`, i.e. a **training-time relative path resolved against the
current working directory**. So either run with CWD = `rxngraphormer/` and place the vocab at
`rxngraphormer/dataset/<NAME>/vocab_smiles.txt`, or patch `parameters.json`'s `data.data_path` to point at
wherever we keep it. `state_dict` keys are normalised through `utils.update_dict_key` (strips DDP
`module.` prefixes), so multi-GPU-trained checkpoints load fine on CPU.

Checkpoint sources (US-004 confirms forward vs retro):

* Reproduction deposit, Figshare article **30498368** (`10.6084/m9.figshare.30498368`) — contains
  `USPTO_480k.7z` (387 MB, forward), `USPTO_STEREO.7z` (392 MB, forward/stereo), `USPTO_50k.7z` (390 MB,
  retro), `USPTO_full.7z` (196 MB, retro), `seq-v2-USPTO_STEREO-20250423_044122_ft.7z` /
  `seq-v2-USPTO_STEREO-20250509_070206_ft.7z` (391 MB each, re-fine-tuned sequence models),
  `pretrained_classification_model.7z`, plus `Test_Dataset.7z` (22 MB — small, good for verification).
  MD5s come from `https://api.figshare.com/v2/articles/30498368`.
* Original authors' deposit: `10.6084/m9.figshare.28356077` (upstream README / `model_path/README.md`).

### Dependency / setup notes

* Third-party imports actually used by the package: `torch`, `torch_geometric`, `torch_scatter`,
  `onmt` (**OpenNMT-py 1.2.0** — beam/greedy search + transformer decoder), `box` (`python-box`),
  `rdkit`, `sklearn`, `pandas`, `numpy`, `tqdm`. `localmapper` / `rxnmapper` are imported **lazily and
  defensively** inside `midgen/midmol.py` (only for delta-mol/mechanism features of the regression
  tasks), so forward/retro inference does not require them — but note `reaction_prediction()` does
  `from rxngraphormer.midgen.midmol import gen_mech_mid_smi` at the top of the function for *all* task
  types, which is another reason to prefer the lower-level path in the backend.
* `requirements_pt221.txt` pins CUDA wheels (`torch==2.2.1+cu121`, `torch_scatter/sparse/cluster
  +pt22cu121`, `dgl==2.1.0`). On this CPU-only box install the CPU equivalents instead:
  `torch` from the PyTorch CPU index, `torch_geometric`, and `torch_scatter` from
  `https://data.pyg.org/whl/torch-<ver>+cpu.html`. `dgl`/`dgllife` are **not** imported by the package.
* Risk to watch in US-005: `OpenNMT-py==1.2.0` historically requires `torchtext==0.4.0`; on torch 2.x a
  `--no-deps` install of OpenNMT-py (plus `configargparse`) may be needed, or only the few `onmt`
  submodules that are imported.
* Upstream expects Python 3.8; our venv is Python 3.11 (`/app/venv`). All vendored `*.py` files parse
  cleanly under 3.11 (checked by `scripts/check_framework.py`).
* Environment observation (2026-08-29): `/app/venv/bin/pip` reports `torch 2.1.2+cpu`, but importing it
  fails with `ModuleNotFoundError: No module named 'torch.torch_version'`; torch lives in
  `/app/user-packages/python/` and looks incomplete — US-002/US-005 must repair/complete that install.
  This is unrelated to US-003 (no torch import is needed for the checks below).

### Quality check for this story

```bash
python scripts/check_framework.py     # static, dependency-free structure + entry-point check
```

Validates the vendored layout, that no nested `.git` remains, that every vendored `*.py` parses, that the
documented entry points exist (AST lookup, no imports), that the four sequence-generation eval configs are
valid JSON, and that repo URL + commit are recorded in `PROJECT_NOTES.md` and `rxngraphormer/VENDOR_INFO.txt`.

### Housekeeping

* The vendored tree keeps its own `.gitignore`, which ignores `model_path/*/`, `dataset/*/`, `save/*`,
  `results/` — handy: dropping checkpoints/datasets inside `rxngraphormer/model_path/…` won't be committed.
  This repo's root `.gitignore` additionally ignores `models/`, `*.7z`, `__pycache__/`, `venv/`, `.env`.
* Never create `.github/workflows/*` in this repo (the available PAT lacks the `workflow` scope).

## Fine-tuned checkpoints — forward/retro mapping (US-004, 2026-08-29)

Downloaded from figshare article 30498368 (API: https://api.figshare.com/v2/articles/30498368), MD5-verified against the figshare manifest:

- `models/seq-v2-USPTO_STEREO-20250423_044122_ft.7z` (id 59201303, md5 163e5a48e4f845eec5e5afd07b00026d)
- `models/seq-v2-USPTO_STEREO-20250509_070206_ft.7z` (id 59201306, md5 52d506a2ecee0c77cad7de03c692f653)
- `data/Test_Dataset.7z` (id 59201294, md5 1d5fcebb2ae10e657180eed52188c9d9) → extracted to `data/Test/`

**Conclusion: BOTH fine-tuned checkpoints are FORWARD-PREDICTION models.** Each
checkpoint's `parameters.json` has `model.task="forward_prediction"` and
`data.data_path="./dataset/USPTO_STEREO"`. Per the framework eval-config mapping
(`uspto_stereo`→FORWARD, `uspto_50k`/`uspto_full`→RETRO), "USPTO_STEREO" = forward.
Checkpoint `20250509` is a continuation training of `20250423` (its training config
`resume_path` points at the `20250423` checkpoint), i.e. a later forward model, NOT a
retro model.

The framework builds the SAME `sequence_generation` architecture for both
`forward-synthesis` and `retro-synthesis` (task_type is inference-time only), but the
released WEIGHTS are forward-trained, so they cannot correctly serve retrosynthesis.

**Correction (US-007, 2026-08-29):** the retro-trained checkpoint IS publicly available —
just in the **original author's** figshare article **28356077** (`10.6084/m9.figshare.28356077`,
"Preprocessed datasets and model weights"), NOT the reproduction article 30498368. The
`model_path/README.md` in the repo points at exactly this article for `USPTO_50k_model.zip`
and `USPTO_full_model.zip`. `USPTO_50k_model.zip` (id 53998184, md5
`1d993b40b8ff38def31788c1ced69de5`) was downloaded, MD5-verified, extracted to
`models/USPTO_50k/`, and its `parameters.json` confirmed `model.task="retrosynthesis"`
(decoder embedding `(75,256)` matches the 75-token `dataset/USPTO_50k/vocab_smiles.txt`).
=> **US-007 (retrosynthesis) is UNBLOCKED and reproduced** via `infer_retro.py` (product
SMILES → top-k precursor sets), all outputs RDKit-valid on 5 USPTO_50k test products.
The reproduction article 30498368 still only has the two forward `_ft` checkpoints; the
retro weights come from 28356077. See the "Retrosynthesis checkpoint — US-007 addendum"
section in MODELS.md for the full table and layout.
