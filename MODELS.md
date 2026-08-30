# Fine-tuned Checkpoints (US-004 + US-012)

Source: figshare article **30498368** "RXNGraphormer Reproduction"
API: `https://api.figshare.com/v2/articles/30498368`

All artifacts below are downloaded with `curl -L -C -` (resume-safe) and the
`.7z` MD5 is verified against the figshare `computed_md5` before extraction.

> **Git hygiene:** `models/` and `*.7z` are listed in `.gitignore` and are
> **NOT committed**. Only this file (`MODELS.md`), `PROJECT_NOTES.md`, and
> `scripts/download_checkpoints.sh` are committed. The large binaries live on
> disk only (mirrored to **GitHub Releases** in US-012).

## GitHub Releases Mirror (US-012)

Checkpoints are mirrored to GitHub Releases for faster, more reliable downloads:

🔗 **Release:** https://github.com/milo0914/rxngraphormer-web/releases/tag/v1.0-weights

| Asset | Task | Size | MD5 | Download URL |
|---|---|---|---|---|
| `forward-stereo-ft.7z` | Forward (USPTO_STEREO) | 391 MB | `52d506a2ecee0c77cad7de03c692f653` | `https://github.com/milo0914/rxngraphormer-web/releases/download/v1.0-weights/forward-stereo-ft.7z` |
| `retro-uspto50k.7z` | Retrosynthesis (USPTO_50k) | 197 MB | `1d993b40b8ff38def31788c1ced69de5` | `https://github.com/milo0914/rxngraphormer-web/releases/download/v1.0-weights/retro-uspto50k.7z` |

**Download scripts** (`backend/download_models.py`, `download_models.sh`) use GitHub Releases as primary source with figshare fallback.

## Fine-tuned checkpoints

| Checkpoint (file) | figshare id | Size (bytes) | MD5 (figshare) | MD5 (verified) | Result |
|---|---|---|---|---|---|
| `seq-v2-USPTO_STEREO-20250423_044122_ft.7z` | 59201303 | 391489512 | `163e5a48e4f845eec5e5afd07b00026d` | `163e5a48e4f845eec5e5afd07b00026d` | ✅ PASS |
| `seq-v2-USPTO_STEREO-20250509_070206_ft.7z` | 59201306 | 391622627 | `52d506a2ecee0c77cad7de03c692f653` | `52d506a2ecee0c77cad7de03c692f653` | ✅ PASS |

Both downloaded into `models/` and extracted (each into its own folder
`models/<name>/`).

### Local extracted layout
```
models/
├── seq-v2-USPTO_STEREO-20250423_044122_ft/      # checkpoint "A" (base fine-tune)
│   ├── parameters.json
│   ├── model/
│   │   ├── valid_checkpoint.pt                  # default ckpt_file for load_pred_model
│   │   └── valid_acc_checkpoint.pt
│   └── log/
└── seq-v2-USPTO_STEREO-20250509_070206_ft/      # checkpoint "B" (later continuation)
    ├── parameters.json
    ├── model/
    │   ├── valid_checkpoint.pt
    │   └── valid_acc_checkpoint.pt
    └── log/
```
To load: pass the folder path (e.g. `models/seq-v2-USPTO_STEREO-20250423_044122_ft`)
to `reaction_prediction(model_path=..., task_type=..., device='cpu')`. The loader
reads `parameters.json` + `model/valid_checkpoint.pt` (default `ckpt_filename`).

## Test set
| File | figshare id | Size (bytes) | MD5 (figshare) | MD5 (verified) | Result |
|---|---|---|---|---|---|
| `Test_Dataset.7z` | 59201294 | 22275360 | `1d5fcebb2ae10e657180eed52188c9d9` | `1d5fcebb2ae10e657180eed52188c9d9` | ✅ PASS |

Extracted into `data/Test/` (213 entries):
`Amide_Coupling_HTE/`, `Amide_Coupling_lit/`, `Meta_C_H/`, `Sulfoxonium/`, plus
`readme.md`. This is the small verification test set used by US-006/US-007.

## Forward vs Retrosynthesis — conclusion (read this carefully)

**Both fine-tuned checkpoints are FORWARD-PREDICTION models.** Neither is a
retrosynthesis checkpoint.

Evidence (from each checkpoint's own `parameters.json`):
- `model.task` = `"forward_prediction"` in **both** checkpoints.
- `data.data_path` = `./dataset/USPTO_STEREO` in **both** checkpoints.
- The framework's eval-config mapping (see `PROJECT_NOTES.md` / US-003) is
  `uspto_stereo → FORWARD` and `uspto_50k` / `uspto_full → RETRO`. The name
  `USPTO_STEREO` therefore confirms forward.
- Checkpoint **B** (`20250509`) is a **continuation training** of checkpoint **A**
  (`20250423`): B's training config contains
  `"resume_path": "./model_path/seq-v2-USPTO_STEREO-20250423_044122_ft/model/valid_acc_checkpoint.pt"`.
  So B is the same forward model taken further, **not** a different task.

### Why one checkpoint cannot serve retro "for free"
The framework (see `rxngraphormer/rxngrapher/eval.py`) builds the **same**
`sequence_generation` architecture for both `forward-synthesis` and
`retro-synthesis` — `task_type` is only an inference-time argument that decides
which file is treated as src vs tgt. The released **weights**, however, are
locked to what the checkpoint was trained on: a forward-trained checkpoint maps
reactants→product. Running it with `task_type="retro-synthesis"` would feed a
product as src and expect reactants as tgt, which the forward weights were never
trained to produce. Correct retrosynthesis therefore requires a
**retro-trained** checkpoint (`USPTO_50k` / `USPTO_full`), which is **not**
present among the figshare fine-tuned releases (the only `*_ft.7z` files are the
two `USPTO_STEREO` forward checkpoints; `USPTO_50k.7z`/`USPTO_full.7z` in the
article are *datasets*, not trained checkpoints).

### Practical mapping for this repo
| Use case | Checkpoint to use | Status |
|---|---|---|
| Forward prediction (US-006) | `models/seq-v2-USPTO_STEREO-20250423_044122_ft` (base) **or** `...20250509_070206_ft` (later) | ✅ available |
| Retrosynthesis (US-007) | retro-trained `USPTO_50k` checkpoint (`models/USPTO_50k`, from figshare **28356077**) | ✅ available — see US-007 addendum below |

**Recommendation:** For US-006 use checkpoint B (`20250509`, the more recent
fine-tune) as the default forward model. US-007 (retrosynthesis) is **UNBLOCKED**:
a genuine retro-trained checkpoint was found in the original author's figshare
article **28356077** (not the reproduction article 30498368, which only has the
two forward `_ft` checkpoints). See the US-007 addendum below.

---

## Retrosynthesis checkpoint — US-007 addendum (figshare article 28356077)

The reproduction article **30498368** only ships forward `_ft` checkpoints (see
"Forward vs Retrosynthesis" above). However, the **original author's** figshare
article **28356077** ("Preprocessed datasets and model weights for RXNGraphormer",
`https://doi.org/10.6084/m9.figshare.28356077`) ships the *actually trained*
backbone model weights referenced by the repo's `model_path/README.md`, including
**retrosynthesis** checkpoints. An exhaustive public search (GitHub repo
`model_path/` listing, the `v1.0.0` release — which has **empty assets**, GitHub
code/repo search for "RXNGraphormer retro" → 0 hits, HuggingFace model search →
empty) found **no other public retro weights**; 28356077 is the canonical source.

### Downloaded retro checkpoint
| Checkpoint (file) | figshare id | Size (bytes) | MD5 (figshare) | MD5 (verified) | Task |
|---|---|---|---|---|---|
| `USPTO_50k_model.zip` | 53998184 | 197722944 | `1d993b40b8ff38def31788c1ced69de5` | `1d993b40b8ff38def31788c1ced69de5` | ✅ PASS — **retrosynthesis** |
| `USPTO_full_model.zip` *(alternative retro ckpt, not downloaded)* | 53995949 | 199742906 | `a00bb7e0d5aea3a809953b955acd9b57` | — | retrosynthesis |

`USPTO_50k_model.zip` was downloaded, MD5-verified, and extracted to
`models/USPTO_50k/` (`parameters.json` + `model/valid_checkpoint.pt`). The
checkpoint's own `parameters.json` declares **`model.task = "retrosynthesis"`**
(confirmed), `data.data_path = ./dataset/USPTO_50k`, `data.vocab_file =
vocab_smiles.txt`. The decoder embedding weight is `(75, 256)` → the decoder
vocabulary has **75 tokens**, matching `dataset/USPTO_50k/vocab_smiles.txt`
(extracted from `USPTO_50k.zip`, id 53991908, MD5 `24af2c6510d72fface1933f042100801`,
75 tokens, `token<TAB>count` format — identical to the USPTO_STEREO vocab format
already used by US-006).

### Local extracted layout
```
models/
└── USPTO_50k/                                      # RETRO checkpoint (from figshare 28356077)
    ├── parameters.json                             # model.task = "retrosynthesis"
    └── model/
        └── valid_checkpoint.pt                      # used by infer_retro.py
dataset/
└── USPTO_50k/
    ├── vocab_smiles.txt                            # 75-token decoder vocab (COMMITTED)
    └── sample_retro_products.txt                   # 5 in-vocab USPTO_50k test products (COMMITTED)
```

### Why this resolves US-007 (and why 30498368 could not)
The framework builds one `sequence_generation` architecture and switches
forward/retro purely via `task_type`; the *weights* decide what it was trained to
do. The 30498368 `_ft` checkpoints have `model.task = "forward_prediction"`, so
they cannot serve retro. The 28356077 `USPTO_50k_model.zip` has `model.task =
"retrosynthesis"`, so feeding a **product** SMILES as src yields the predicted
**precursor set** — a genuine retrosynthesis. `infer_retro.py` wraps exactly this
(`task_type="retro-synthesis"`, `device='cpu'`), reusing the proven loader from
US-006. Verified on 5 USPTO_50k test products: all 13 reported precursor-set
candidates were RDKit-valid (see `sample_retro_outputs.txt`).

> Note: the 28356077 `USPTO_STEREO_model.zip` / `USPTO_480k_model.zip` are
> forward checkpoints (same role as the 30498368 `_ft` ones); only `USPTO_50k_*`
> and `USPTO_full_*` are retro. `buchwald_hartwig` / `C_H_func` / `suzuki_miyaura`
> / `thiol_addition` / `external_validation` are single-step task models.

## Reproduce
See `scripts/download_checkpoints.sh` (uses `curl -C -` resume + `md5sum`
verification, no large files committed).
