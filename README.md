# SPARK: Input-Conditioned Sparse Activation Modulation for Frozen DiT-based Super-Resolution

Reference implementation of SPARK on the **TSD-SR** backbone, covering the main
comparison table.

DiT-based SR backbones concentrate most of their activation energy in a handful
of *massive* channels. SPARK exploits that sparse subspace as an adaptation
interface: it identifies the dominant channels of every block and stream, then
trains a small input-conditioned predictor that emits bounded per-channel affine
parameters (scale γ, shift β) for **only those channels**. The DiT backbone and
the VAE stay frozen; the predictor is the only thing optimized.

The paper evaluates three backbones (TSD-SR, DiT4SR, TEASR). This repository
covers **TSD-SR only** — the pipeline is identical for the others, so it is not
duplicated here.

> **Anonymous submission.** No author, institution or cluster information is
> present in this repository. All machine-specific paths were replaced by
> environment variables (see [Configuration](#configuration)).

---

## Method

**Phase 1 — online channel selection (Alg. 1).** Mini-batches of the optimization
set are streamed through the frozen backbone with no gradients. For each block ℓ
and stream s ∈ {hidden, encoder}, the per-channel activation magnitude is the mean
absolute activation over tokens and batch (Eq. 1); an EMA smooths it across
iterations (Eq. 3, λ = 0.95). Every W = 5 updates the provisional top-K set is
snapshotted, and once P = 4 consecutive window transitions have mean Jaccard
overlap ≥ τ = 0.9 for **every** block and stream, the scan terminates early
(Eq. 5). K = 8 channels are kept per stream and per block.

This is a *partial* pass: it normally converges after ≈400 images (≈25
mini-batches of 16), roughly 8 minutes on one GPU, instead of a full sweep.

**Phase 2 — image-conditioned predictor.** The LR image is encoded by the same
frozen VAE the SR pipeline already uses; the latent is spatially pooled into a
compact vector and passed to an MLP (two hidden layers of width 256, SiLU) whose
output dimension is 2·Σ|I| — one (γ, β) pair per selected channel, per stream, per
block (Eq. 9). A sigmoid plus linear rescaling bounds them to γ ∈ [0.5, 1.5] and
β ∈ [−0.2, 0.2] (Eq. 10), which the paper's ablation shows is what keeps the
modulation from degenerating into high-frequency texture. Gradients flow through
the frozen backbone back to the predictor only.

The objective (Eq. 11) is

```
L = LPIPS(ŷ, y) − α_LIQE · LIQE(ŷ) + α_TV · L_TV(ŷ),    α_LIQE = 0.1, α_TV = 1e-4
```

Trainable parameters: a few hundred thousand, orders of magnitude below the
backbone.

---

## Contents

| Path | What it does |
|---|---|
| `utils/phase1_online.py` | **Phase 1** — Algorithm 1: online EMA ranking with Jaccard stability termination |
| `train_adaln_predictor.py` | Driver: runs Phase 1, then trains the Phase-2 predictor |
| `utils/adaln_utils.py` | Capture hooks, modulation hooks, the predictor module, the SR loop, the full argument surface |
| `infer_adaln_predictor_dataset.py` | Benchmark inference **with** the trained predictor → the `+ SPARK` rows |
| `infer_tsdsr_baseline_dataset.py` | Benchmark inference of the **frozen TSD-SR** → the baseline rows |
| `models/autoencoder_kl.py` | SD3 VAE with the tiled encode/decode TSD-SR uses |
| `utils/{vaehook,wavelet_color_fix,util,device}.py` | Tiled VAE, wavelet/AdaIN colour fix, LoRA loading, device helpers |
| `eval/compute_metrics.py` | Standalone metric evaluation over any pair of image folders |
| `eval/infer_tsdsr_single.py` | Frozen TSD-SR on a folder of images (sanity check / qualitative figures) |
| `scripts/*.sh` | Thin wrappers pinning the paper configuration |

---

## Installation (uv)

Pinned to the exact versions the reported numbers were produced with:
**CPython 3.11**, torch 2.2.2 + cu121, diffusers 0.29.1, peft 0.15.0, pyiqa 0.1.15.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # install uv (once)

cd spark
export UV_CACHE_DIR=$PWD/.uv-cache                  # uv's cache can reach several GB
# On a network filesystem (NFS/Lustre/BeeGFS) also set this, or uv's hardlinking
# fails with "Device or resource busy" while building source distributions:
# export UV_LINK_MODE=copy

uv venv --python 3.11
uv sync

source .venv/bin/activate
python -c "import torch, diffusers, pyiqa; print(torch.__version__, torch.cuda.is_available())"
```

`pyproject.toml` already handles the two awkward pins:

* `torch==2.2.2` / `torchvision==0.17.2` come from the `download.pytorch.org/whl/cu121`
  index, declared as an explicit `[[tool.uv.index]]` — they are not on PyPI.
* `opencv-python-headless` declares `numpy>=2` but runs correctly against the
  `numpy==1.26.4` that torch 2.2.2 and pyiqa need, so `[tool.uv] override-dependencies`
  forces 1.26.4 instead of letting the resolver fail.

`setuptools==65.5.0` is pinned on purpose: setuptools ≥ 80 removed `pkg_resources`,
which `peft`'s `BaseTunerLayer` still imports.

**Different CUDA version:** change the index URL in `pyproject.toml` (e.g.
`.../whl/cu118`) and pick the matching torch build. SPARK has no CUDA-specific code.

### Hardware

One **48 GB** GPU (all reported runs used an NVIDIA L40S). On TSD-SR: Phase 1
≈8 minutes, Phase 2 ≈2.5 hours, DIV2K inference ≈30 minutes, DRealSR/RealSR
evaluation ≈5 minutes. Phase 1's memory overhead is negligible — only per-channel
activation means are stored.

Add `--is_use_tile` to any inference command for the tiled VAE / tiled latent path
if you are memory constrained or processing images larger than 512×512.

---

## Pretrained weights

Two sets of weights must be downloaded; neither is redistributed here.

1. **Stable Diffusion 3 Medium** (diffusers layout) —
   `stabilityai/stable-diffusion-3-medium-diffusers` on HuggingFace. Gated: accept
   the licence, then
   `huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers --local-dir checkpoints/stable-diffusion-3-medium-diffusers`.
2. **TSD-SR LoRA weights + prompt embeddings** — from the official TSD-SR release
   (Dong et al., CVPR 2025). Use the `checkpoint/tsdsr-mse` LoRA and the
   `dataset/default` prompt embeddings.

```
checkpoints/
├── stable-diffusion-3-medium-diffusers/   # SD3, diffusers layout
└── tsdsr/
    ├── lora/                              # TSD-SR LoRA (checkpoint/tsdsr-mse)
    └── embeddings/                        # prompt embeddings (dataset/default)
```

The scripts never download anything: `train_adaln_predictor.py` sets
`HF_HUB_OFFLINE=1`. Export `HF_ALLOW_DOWNLOAD=1` if you do want it to fetch
missing files.

---

## Data

The predictor is trained on the **DIV2K training set** with synthetic
degradations from the Real-ESRGAN pipeline. Evaluation uses DIV2K validation
plus the real-world DRealSR and RealSR benchmarks; for the real-world sets each
LR image is center-cropped to 128×128 with the corresponding aligned HR region.

```
datasets/
├── DRealSR/          test_LR/  test_HR/      #   93 pairs
├── RealSR/           test_LR/  test_HR/      #  100 pairs
├── DIV2K/            lr/       gt/           # 3000 pairs (validation)
└── DIV2K_train/      LR/       HR/           # 512x512 training crops + degraded LR
```

Folder names matter: `DRealSR` and `RealSR` are read from `test_LR`/`test_HR`,
`DIV2K` from `lr`/`gt`. Override any of them with `--input_dir` / `--gt_dir`.

---

## Configuration

Instead of hard-coded paths, every default is read from an environment variable.
Edit `scripts/env.sh` once and source it:

```bash
source scripts/env.sh
```

| Variable | Meaning | Also a CLI flag |
|---|---|---|
| `SD3_MODEL_PATH` | SD3 Medium (diffusers layout) | `--pretrained_model_name_or_path` |
| `TSDSR_LORA_DIR` | TSD-SR LoRA weights | `--lora_dir` |
| `TSDSR_EMBEDDING_DIR` | TSD-SR prompt embeddings | `--embedding_dir` |
| `DATA_ROOT` | root holding the dataset folders above | `--input_dir` / `--gt_dir` |
| `OUTPUT_ROOT` | where runs and evaluations are written | `--output_dir` |

Flags always win over environment variables.

---

## Reproducing the main table

### 1. Frozen baseline

```bash
source scripts/env.sh
bash scripts/eval_baseline.sh DRealSR
bash scripts/eval_baseline.sh RealSR
bash scripts/eval_baseline.sh DIV2K
```

Each run writes into `$OUTPUT_ROOT/baseline/<dataset>`:

```
images/                                 # the SR outputs
reports/baseline_metrics_per_image.csv  # one row per image
reports/baseline_metrics_summary.json   # mean/std per metric -- the table cells
```

`--resume` is on, so an interrupted run picks up where it stopped.

### 2. Train SPARK

```bash
bash scripts/train_predictor.sh topk8_paper
```

Both phases run inside this one command. Paper configuration:

| Setting | Value | Flag |
|---|---|---|
| selected channels K | 8 per stream **and** per block | `--topk 8` |
| Phase-1 procedure | online EMA + stability termination | `--phase1_selector online_ema` |
| Phase-1 importance | mean absolute activation (Eq. 1) | `--phase1_importance_mode mean_abs` |
| Phase-1 mini-batch | 16 | `--phase1_batch_size 16` |
| EMA decay λ | 0.95 | `--phase1_ema_decay 0.95` |
| window W / transitions P | 5 / 4 | `--phase1_window 5 --phase1_window_transitions 4` |
| stability threshold τ | 0.9 | `--phase1_stability_tau 0.9` |
| modulation bounds | γ ∈ [0.5, 1.5], β ∈ [−0.2, 0.2] | `--adaln_gamma_{min,max} --adaln_beta_{min,max}` |
| predictor | MLP, 2 hidden layers × 256, SiLU | `--adaln_predictor_hidden_dim 256 --adaln_predictor_num_layers 3` |
| loss | `LPIPS − 0.1·LIQE + 1e-4·TV` | `--adaln_predictor_alpha_{lpips,liqe,tv}` |
| optimisation | Adam, lr 1e-4, one epoch, effective batch 16 | `--adaln_predictor_lr 1e-4`, batch 4 × grad-accum 4 |
| model selection | validate every 100 steps, early stop after 5 | `--adaln_predictor_val_every 100 --adaln_predictor_early_stop_patience 5` |

`--phase1_max_samples` is an **upper budget** for Phase 1, not a target. The scan
ends either when the stability criterion fires or when the budget runs out;
either way the selection is the top-K of the final EMA scores. The
`[phase1-ema]` line reports which happened, and how many images and windows were
actually consumed:

```
[phase1-ema] stability criterion met (min mean Jaccard 0.944 >= tau 0.9); images=400 ...
[phase1-ema] exhausted optimization set; images=640 mini-batches=40 windows=8 converged=False
```

Note that the criterion is strict: `tau` must be met by **every** block and
stream simultaneously, and with K = 8 the per-transition Jaccard is quantized to
1.0 (identical sets), 0.778 (one channel differs), 0.600 (two differ), so a mean
of 0.9 over P = 4 transitions allows at most a single one-channel change. If it
does not fire on your data, either give it a larger budget or relax
`--phase1_stability_tau` / raise `--phase1_window`.

The run directory ends up as:

```
outputs/topk8_paper/
└── reports/
    ├── adaln_predictor.pt      # the trained predictor (the only trained weights)
    ├── adaln_predictor.json    # selected channels, layout, config, val history
    └── validation_*.csv
```

Everything the evaluation needs is inside `reports/`, so no hyper-parameter has to
be repeated at inference time.

Ablation knobs, all overridable from the environment:

```bash
TOPK=16 bash scripts/train_predictor.sh topk16              # effect of K (Fig. 5)
SELECTION_MODE=random bash scripts/train_predictor.sh rnd    # Tab. 3 controls
SELECTION_MODE=bottomk bash scripts/train_predictor.sh bot
IMPORTANCE_MODE=std bash scripts/train_predictor.sh std      # supp. C.2
```

### 3. Evaluate SPARK

```bash
bash scripts/eval_predictor.sh outputs/topk8_paper DRealSR
bash scripts/eval_predictor.sh outputs/topk8_paper RealSR
bash scripts/eval_predictor.sh outputs/topk8_paper DIV2K
```

Each run writes the same structure, under the `predictor_` prefix:

```
outputs/topk8_paper/predictor_eval_<dataset>/
├── images/
└── reports/
    ├── predictor_metrics_per_image.csv
    └── predictor_metrics_summary.json   # the table cells
```

### 4. Metrics

Both inference scripts report the paper's metrics — SSIM, LPIPS, MANIQA, MUSIQ,
CLIP-IQA, TOPIQ, LIQE — via `pyiqa`. All are reported ×100 in the paper's tables.
LIQE is also part of the training objective, so it is not an independent measure.

**One convention matters for comparability:** SSIM (and PSNR) are computed on the
**Y channel in YCbCr**, `pyiqa.create_metric("ssim", test_y_channel=True,
color_space="ycbcr")`. RGB SSIM differs from Y-channel SSIM by roughly 3 points on
these benchmarks, so baseline and method must be scored the same way. The
convention is identical in `utils/adaln_utils.py`, both inference scripts and
`eval/compute_metrics.py`; do not mix in numbers computed differently.

To re-score existing output folders without re-running the model:

```bash
python eval/compute_metrics.py \
  --inp_imgs outputs/baseline/DRealSR/images \
  --gt_imgs  datasets/DRealSR/test_HR \
  --log      logs/metrics
```

Point `--inp_imgs` at the `images/` directory, not at the run directory, and note
that this script asserts the two folders hold the **same number of images** -- it
cannot score a partial run. For those, use the dataset scripts' own
`--metrics_only --recompute_metrics` instead.

or, for the dataset scripts, `--metrics_only --recompute_metrics`.

---

## Qualitative results

```bash
python eval/infer_tsdsr_single.py \
  --pretrained_model_name_or_path "$SD3_MODEL_PATH" \
  --lora_dir "$TSDSR_LORA_DIR" \
  --embedding_dir "$TSDSR_EMBEDDING_DIR" \
  -i path/to/images -o outputs/demo \
  --align_method wavelet
```

---

## Notes and caveats

* **Phase 1 must be re-run per backbone.** Channel selection is backbone- and
  checkpoint-specific; changing either invalidates the selected map.
* **Determinism.** Seeds are set (`--seed 42`), but the diffusion step and the IQA
  backbones run in `bf16` on GPU, so the last decimal moves between machines.
  The paper reports a mean cross-seed standard deviation of 0.0021 on
  scale-normalized metrics; differences of ≲0.05 raw points are noise.
* **Frozen backbone.** The SD3 transformer, the TSD-SR LoRA and the VAE are never
  updated. Only the predictor is optimized.
* **Argument surface.** `utils/adaln_utils.py::parse_args` exposes many more flags
  than SPARK uses (alternative scoring modes, gating variants, test-time
  optimization, a `--phase1_selector full_pass` fallback that ranks after a
  complete sweep instead of terminating early). `scripts/train_predictor.sh` pins
  the ones the paper reports; the rest are left at their defaults.

---

## Acknowledgements

Built on **TSD-SR** (Dong et al., *TSD-SR: One-Step Diffusion with Target Score
Distillation for Real-World Image Super-Resolution*, CVPR 2025) and its released
weights. `models/autoencoder_kl.py`, `utils/vaehook.py`,
`utils/wavelet_color_fix.py` and `eval/` are adapted from the TSD-SR / OSEDiff
codebases. Evaluation uses [pyiqa](https://github.com/chaofengc/IQA-PyTorch).
Test sets are the StableSR benchmark crops.

## Licence

Apache 2.0, inherited from TSD-SR — see [`LICENSE`](LICENSE).

## Citation

Withheld for anonymous review.
