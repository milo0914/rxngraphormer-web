# Fine-tuned Checkpoints (US-004)

Source: figshare article **30498368** "RXNGraphormer Reproduction"
API: `https://api.figshare.com/v2/articles/30498368`

All artifacts below are downloaded with `curl -L -C -` (resume-safe) and the
`.7z` MD5 is verified against the figshare `computed_md5` before extraction.

> **Git hygiene:** `models/` and `*.7z` are listed in `.gitignore` and are
> **NOT committed**. Only this file (`MODELS.md`), `PROJECT_NOTES.md`, and
> `scripts/download_checkpoints.sh` are committed. The large binaries live on
> disk only (they will later be mirrored to GitHub Releases in US-012).

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
| Retrosynthesis (US-007) | retro-trained `USPTO_50k`/`USPTO_full` checkpoint | ⚠️ **NOT in this figshare article** — blocked until a retro checkpoint is sourced/trained |

**Recommendation:** For US-006 use checkpoint B (`20250509`, the more recent
fine-tune) as the default forward model. US-007 (retrosynthesis) is currently
**blocked** on obtaining a retro-trained checkpoint; flag this to the controller
before starting US-007.

## Reproduce
See `scripts/download_checkpoints.sh` (uses `curl -C -` resume + `md5sum`
verification, no large files committed).
