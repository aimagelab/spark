import argparse
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import random
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import csv
from peft import LoraConfig
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
import math
from models.autoencoder_kl import AutoencoderKL
from utils.vaehook import _init_tiled_vae
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from utils.util import load_lora_state_dict
import json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
PHASE1_NR_METRICS = {"liqe", "maniqa", "musiq"}
FR_IQA_METRICS = {"lpips", "dists", "psnr", "ssim", "dinov2_cosine"}
LOWER_BETTER_METRICS = {"lpips", "dists", "niqe"}


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    value = str(v).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=False, default=os.environ.get("SD3_MODEL_PATH", "checkpoints/stable-diffusion-3-medium-diffusers"))
    parser.add_argument("--lora_dir", type=str, default=os.environ.get("TSDSR_LORA_DIR", "checkpoints/tsdsr/lora"))
    parser.add_argument("--embedding_dir", type=str, default=os.environ.get("TSDSR_EMBEDDING_DIR", "checkpoints/tsdsr/embeddings"))
    parser.add_argument("--input_dir", type=str, default=os.path.join(os.environ.get("DATA_ROOT", "datasets"), "DRealSR", "test_LR"))
    parser.add_argument("--gt_dir", type=str, default=os.path.join(os.environ.get("DATA_ROOT", "datasets"), "DRealSR", "test_HR"))
    parser.add_argument("--opt_input_dir", type=str, default=os.path.join(os.environ.get("DATA_ROOT", "datasets"), "DIV2K_train", "LR"))
    parser.add_argument("--opt_gt_dir", type=str, default=os.path.join(os.environ.get("DATA_ROOT", "datasets"), "DIV2K_train", "HR"))
    parser.add_argument("--output_dir", "-o", type=str, default="outputs/probe_tsdsr")

    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--rank_vae", type=int, default=64)

    parser.add_argument("--is_use_tile", type=bool, default=False)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32", "bf16"], default="bf16")
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="wavelet")

    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--selection_mode", type=str, choices=["topk", "bottomk", "middlek", "random"], default="topk")
    parser.add_argument(
        "--layer_scope",
        type=str,
        choices=["all", "early", "mid", "late"],
        default="all",
        help=(
            "Transformer block scope where channel modulation/ablation is applied: "
            "all blocks, early third, middle third, or late third."
        ),
    )
    parser.add_argument(
        "--phase1_importance_mode",
        type=str,
        choices=["std", "mean_abs", "kurtosis", "snr_proxy", "fused_stats", "gradxact"],
        default="mean_abs",
        help=(
            "Channel importance used in phase-1 aggregation: "
            "'mean_abs' uses mean absolute activation, 'std' uses activation std, "
            "'kurtosis' uses excess kurtosis, 'snr_proxy' uses |mean|/std, "
            "'fused_stats' combines robust-normalized explainable stats, "
            "and 'gradxact' uses mean(abs(dL/dh * h))."
        ),
    )
    # --- Phase 1, online EMA selection (Algorithm 1) --------------------------
    parser.add_argument(
        "--phase1_selector",
        type=str,
        choices=["online_ema", "full_pass"],
        default="online_ema",
        help=(
            "Channel-selection procedure. 'online_ema' is the method described in "
            "the paper: an EMA of per-channel activation magnitude updated per "
            "mini-batch, terminated early once the top-K ranking is stable. "
            "'full_pass' scores every image once and ranks at the end."
        ),
    )
    parser.add_argument(
        "--phase1_batch_size",
        type=int,
        default=16,
        help="Mini-batch size of the Phase-1 scan (online_ema only).",
    )
    parser.add_argument(
        "--phase1_ema_decay",
        type=float,
        default=0.95,
        help="EMA decay lambda in Eq. 3 (online_ema only).",
    )
    parser.add_argument(
        "--phase1_window",
        type=int,
        default=5,
        help="Window length W: snapshot the top-K ranking every W updates.",
    )
    parser.add_argument(
        "--phase1_window_transitions",
        type=int,
        default=4,
        help="Number of consecutive window transitions P used by the stability test.",
    )
    parser.add_argument(
        "--phase1_stability_tau",
        type=float,
        default=0.9,
        help=(
            "Stability threshold tau in Eq. 5. The scan stops once the mean Jaccard "
            "overlap over the last P window transitions is >= tau for every block "
            "and stream."
        ),
    )
    parser.add_argument(
        "--phase1_channel_agg",
        type=str,
        choices=["mean", "median", "trimmed_mean"],
        default="mean",
        help=(
            "How to aggregate per-image channel scores before top-k selection. "
            "Use median/trimmed_mean for better robustness to outliers."
        ),
    )
    parser.add_argument(
        "--phase1_trim_fraction",
        type=float,
        default=0.1,
        help=(
            "Trim fraction per side used when --phase1_channel_agg=trimmed_mean. "
            "Values outside [0, 0.49] are clamped during execution."
        ),
    )
    parser.add_argument(
        "--phase1_cross_image_metric",
        type=str,
        choices=["std", "mad", "iqr"],
        default="mad",
        help=(
            "Cross-image specificity statistic used by --phase1_importance_mode=fused_stats. "
            "mad and iqr are more robust than std."
        ),
    )
    parser.add_argument("--phase1_fused_weight_mean_abs", type=float, default=1.0)
    parser.add_argument("--phase1_fused_weight_kurtosis", type=float, default=0.5)
    parser.add_argument("--phase1_fused_weight_snr", type=float, default=0.5)
    parser.add_argument("--phase1_fused_weight_cross_image", type=float, default=1.0)
    parser.add_argument("--phase1_fused_weight_std_penalty", type=float, default=0.0)
    parser.add_argument(
        "--phase1_gradxact_mode",
        type=str,
        choices=["abs_grad_act", "grad_only", "normalized_grad_act", "integrated_gradients"],
        default="normalized_grad_act",
        help=(
            "Scoring mode used by phase1 gradxact: abs_grad_act (legacy), "
            "grad_only, normalized_grad_act (recommended), or integrated_gradients (slower, more precise)."
        ),
    )
    parser.add_argument(
        "--phase1_ig_steps",
        type=int,
        default=8,
        help=(
            "Number of interpolation steps used when --phase1_gradxact_mode=integrated_gradients. "
            "Higher values are slower but usually more stable/precise."
        ),
    )
    parser.add_argument(
        "--phase1_ig_baseline",
        type=str,
        choices=["zero"],
        default="zero",
        help="Baseline used for Integrated Gradients in phase-1.",
    )
    parser.add_argument(
        "--phase1_loss_norm_mode",
        type=str,
        choices=["none", "loss", "loss_sqrt"],
        default="loss",
        help=(
            "Normalize phase1 gradients per sample before channel scoring: "
            "none, divide by |loss|, or divide by sqrt(|loss|)."
        ),
    )
    parser.add_argument(
        "--phase1_grad_clip_norm",
        type=float,
        default=5.0,
        help="Per-capture gradient norm clip in phase1 gradxact (<=0 disables).",
    )
    parser.add_argument(
        "--phase1_grad_eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon used by phase1 gradxact normalization/clipping.",
    )
    parser.add_argument(
        "--phase1_grad_diag_every",
        type=int,
        default=25,
        help="Print phase1 grad diagnostics every N samples (<=0 disables).",
    )
    parser.add_argument(
        "--phase1_grad_loss",
        type=str,
        choices=["l1", "l2", "l1_l2", "lpips", "liqe", "maniqa", "musiq", "hybrid"],
        default="l1",
        help=(
            "Loss used by phase1_importance_mode=gradxact. "
            "Supports reconstruction losses, NR-IQA losses, or hybrid."
        ),
    )
    parser.add_argument(
        "--phase1_hybrid_recon_loss",
        type=str,
        choices=["l1", "l2", "l1_l2", "lpips"],
        default="l1",
        help="Reconstruction term used when --phase1_grad_loss=hybrid.",
    )
    parser.add_argument(
        "--phase1_hybrid_nr_metric",
        type=str,
        choices=["liqe", "maniqa", "musiq"],
        default="liqe",
        help="No-reference metric used when --phase1_grad_loss=hybrid.",
    )
    parser.add_argument(
        "--phase1_hybrid_recon_weight",
        type=float,
        default=1.0,
        help="Weight of reconstruction term when --phase1_grad_loss=hybrid.",
    )
    parser.add_argument(
        "--phase1_hybrid_nr_weight",
        type=float,
        default=1.0,
        help="Weight of NR-IQA term when --phase1_grad_loss=hybrid.",
    )
    parser.add_argument(
        "--topk_cache_file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file with cached phase1 topk results. "
            "Accepts either adaln_global.json (uses phase1_topk/phase1_bottomk) "
            "or a raw topk map keyed by block index."
        ),
    )
    parser.add_argument(
        "--phase1_max_samples",
        type=int,
        default=0,
        help=(
            "Maximum number of samples used in phase-1 topk aggregation from --opt_input_dir. "
            "<=0 uses all samples. This does not limit phase-2 AdaLN optimization data."
        ),
    )
    parser.add_argument(
        "--phase1_shuffle_samples",
        action="store_true",
        help="Randomly sample phase-1 subset (otherwise uses the first N files after sorting).",
    )
    parser.add_argument(
        "--zero_out_selection_mode",
        type=str,
        choices=["none", "topk", "bottomk", "middlek", "random"],
        default="none",
        help=(
            "Channel zero-ablation mode used during inference. "
            "'topk' zeroes selected top-k channels, 'bottomk' zeroes bottom-k channels, "
            "'middlek' zeroes middle-ranked channels, 'random' zeroes random-k channels, "
            "and 'none' disables zero-ablation."
        ),
    )
    # Deprecated alias kept for backward compatibility.
    parser.add_argument("--bottomk_zero_out", action="store_true")
    parser.add_argument("--modulate_hidden", type=_str2bool, default=True, help="Enable/disable hidden-state stream modulation (true/false).",)
    parser.add_argument("--modulate_encoder", type=_str2bool, default=True, help="Enable/disable encoder-hidden-state stream modulation (true/false).",)
    parser.add_argument(
        "--stream_weight_mode",
        type=str,
        choices=[
            "manual",
            "learned_gate",
            "learned_gate_softmax2",
            "learned_gate_sigmoid2",
            "learned_gate_tristate",
            "learned_gate_hard_global_ste",
            "learned_gate_std_prior_offline",
            "liqe_best3_forward",
        ],
        default="manual",
        help=(
            "How hidden/encoder streams are combined. "
            "manual uses --modulate_hidden/--modulate_encoder toggles; "
            "learned_gate or learned_gate_softmax2 learns competing 2-way softmax weights (sum=1); "
            "learned_gate_sigmoid2 learns independent sigmoid weights (both can be near 1); "
            "learned_gate_tristate learns 3 states: hidden-only, encoder-only, or both; "
            "learned_gate_hard_global_ste learns one global hs/ehs selection with a straight-through hard gate; "
            "learned_gate_std_prior_offline picks hs/ehs/both from phase-1 std ratios per block; "
            "liqe_best3_forward runs hs-only, ehs-only, and both at inference and picks best LIQE."
        ),
    )
    parser.add_argument(
        "--stream_gate_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for learned stream gates (lower => sharper decisions).",
    )
    parser.add_argument(
        "--stream_gate_init",
        type=str,
        choices=["neutral", "hs", "ehs"],
        default="neutral",
        help="Initialization prior for learned stream gate logits.",
    )
    parser.add_argument(
        "--stream_gate_init_scale",
        type=float,
        default=2.0,
        help="Logit magnitude used by --stream_gate_init when not neutral.",
    )
    parser.add_argument(
        "--stream_prior_ratio_low",
        type=float,
        default=0.8,
        help="Offline std-prior: ratio below this selects hs.",
    )
    parser.add_argument(
        "--stream_prior_ratio_high",
        type=float,
        default=1.2,
        help="Offline std-prior: ratio above this selects ehs.",
    )
    parser.add_argument(
        "--stream_prior_eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for offline std-prior ratio computation.",
    )
    parser.add_argument(
        "--liqe_best3_opt_warmup_steps",
        type=int,
        default=2,
        help=(
            "When --stream_weight_mode=liqe_best3_forward, run this many TTO optimization "
            "steps for each candidate stream (hs, ehs, both) before selection."
        ),
    )
    parser.add_argument(
        "--liqe_best3_opt_refine_steps",
        type=int,
        default=3,
        help=(
            "When --stream_weight_mode=liqe_best3_forward, run this many additional TTO "
            "steps after selecting the best stream from warmup."
        ),
    )

    parser.add_argument("--adaln_optimize", action="store_true")
    #parser.add_argument("--adaln_steps", type=int, default=10000)
    parser.add_argument("--adaln_batch_size", type=int, default=8, help="Number of optimization samples used per AdaLN step.",)
    parser.add_argument(
        "--adaln_grad_accum_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation micro-steps per AdaLN optimizer step.",
    )
    parser.add_argument("--adaln_lr", type=float, default=1e-2)
    parser.add_argument("--adaln_loss", type=str, choices=["l1", "l2", "l1_l2", "liqe", "lpips_liqe"], default="l1")
    parser.add_argument("--adaln_lpips_weight", type=float, default=4.0, help="Weight of LPIPS term when --adaln_loss=lpips_liqe.",)
    parser.add_argument("--adaln_liqe_weight", type=float, default=1.0, help="Weight of LIQE term when --adaln_loss=lpips_liqe.",)
    parser.add_argument(
        "--adaln_balance_lpips_liqe",
        type=_str2bool,
        default=True,
        help=(
            "Balance LPIPS/LIQE gradients by per-term global grad norm before applying weights "
            "(recommended for --adaln_loss=lpips_liqe, especially at larger --topk)."
        ),
    )
    parser.add_argument(
        "--adaln_balance_eps",
        type=float,
        default=1e-6,
        help="Epsilon used in LPIPS/LIQE gradient-norm balancing.",
    )
    parser.add_argument("--adaln_gamma_min", type=float, default=0.5)
    parser.add_argument("--adaln_gamma_max", type=float, default=1.5)
    parser.add_argument("--adaln_beta_min", type=float, default=-0.2)
    parser.add_argument("--adaln_beta_max", type=float, default=0.2)
    parser.add_argument(
        "--adaln_free_params",
        action="store_true",
        help=(
            "Use unconstrained AdaLN gamma/beta parameters (no sigmoid-range projection). "
            "When enabled, gamma initializes to 1 and beta to 0."
        ),
    )
    parser.add_argument(
        "--adaln_modulation_type",
        type=str,
        choices=["affine", "scale", "shift", "residual"],
        default="affine",
        help=(
            "Modulation parameterization for top-k channels. "
            "'affine' uses gamma/beta, 'scale' uses only gamma, 'shift' uses only beta, "
            "and 'residual' learns a projection and adds it back (x + P(x))."
        ),
    )
    parser.add_argument(
        "--adaln_per_channel",
        action="store_true",
        help=(
            "When set, use independent per-channel parameters. "
            "For residual mode this is a diagonal projection; otherwise a full topk->topk projection is used."
        ),
    )
    parser.add_argument("--adaln_val_every", type=int, default=20)
    parser.add_argument("--adaln_val_max_samples", type=int, default=100)
    parser.add_argument("--adaln_predictor_train", action="store_true")
    parser.add_argument("--adaln_predictor_ckpt", type=str, default="")
    parser.add_argument("--adaln_predictor_steps", type=int, default=1000)
    parser.add_argument(
        "--adaln_predictor_max_num_steps",
        type=int,
        default=None,
        help=(
            "Optional explicit max optimizer steps for predictor training. "
            "When set, training stops at the first limit reached between this and --adaln_predictor_steps."
        ),
    )
    parser.add_argument(
        "--adaln_predictor_max_num_epochs",
        type=int,
        default=None,
        help=(
            "Optional max number of predictor-training epochs over optimization samples. "
            "Training stops when the first limit is reached between --adaln_predictor_steps and this value."
        ),
    )
    parser.add_argument("--adaln_predictor_batch_size", type=int, default=4)
    parser.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        default=True,
        help="Enable batched/parallel AdaLN predictor training within each optimizer step.",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable batched/parallel AdaLN predictor training within each optimizer step.",
    )
    parser.add_argument(
        "--adaln_predictor_grad_accum_steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer step in predictor training.",
    )
    parser.add_argument("--adaln_predictor_lr", type=float, default=1e-3)
    parser.add_argument("--adaln_predictor_hidden_dim", type=int, default=256)
    parser.add_argument("--adaln_predictor_num_layers", type=int, default=3)
    parser.add_argument("--adaln_predictor_dropout", type=float, default=0.0)
    parser.add_argument(
        "--adaln_predictor_feature_type",
        type=str,
        choices=["mean", "mean_std"],
        default="mean_std",
        help="VAE latent summary used to condition the AdaLN predictor.",
    )
    parser.add_argument("--adaln_predictor_val_every", type=int, default=50)
    parser.add_argument("--adaln_predictor_val_max_samples", type=int, default=93)
    parser.add_argument(
        "--adaln_predictor_alpha_lpips",
        type=float,
        default=0.0,
        help="Weight of LPIPS loss term for AdaLN predictor training.",
    )
    parser.add_argument(
        "--adaln_predictor_alpha_liqe",
        type=float,
        default=1.0,
        help="Weight of LIQE loss term for AdaLN predictor training.",
    )
    parser.add_argument(
        "--adaln_predictor_iqa_metric",
        type=str,
        default="liqe",
        choices=["liqe", "maniqa", "maniqa-pipal", "musiq", "topiq_nr", "clipiqa"],
        help=(
            "No-reference IQA metric used as the perceptual training objective. "
            "All choices are higher-is-better and enter the loss as -metric(sr). "
            "Default 'liqe' reproduces the original objective exactly."
        ),
    )
    parser.add_argument(
        "--adaln_predictor_maniqa_crops",
        type=int,
        default=1,
        help=(
            "Crops per image when MANIQA is the training objective. pyiqa's eval "
            "path uses 20 and keeps all of them in the autograd graph (OOM on a "
            "45G card); MANIQA's own training path uses 1. Evaluation is unaffected."
        ),
    )
    parser.add_argument(
        "--adaln_predictor_alpha_iqa",
        type=float,
        default=None,
        help=(
            "Weight of the IQA objective term. The metrics live on very different "
            "numeric scales, so this compensates for that: LIQE ~4.5 (alpha 0.1), "
            "MANIQA ~0.60 (alpha 0.5), MUSIQ ~68 (alpha 0.005), TOPIQ ~0.67 "
            "(alpha 0.5), all contributing ~0.3-0.45 to the loss. "
            "Defaults to --adaln_predictor_alpha_liqe when unset."
        ),
    )
    parser.add_argument(
        "--adaln_predictor_maniqa_test_sample",
        type=int,
        default=20,
        help=(
            "Number of uniform 224px crops MANIQA averages per image. pyiqa's "
            "default is 20, which keeps 20 ViT graphs per image alive in the "
            "backward pass and OOMs a 45G GPU; lower it when using MANIQA as a "
            "training objective. Ignored for the other metrics."
        ),
    )
    parser.add_argument(
        "--adaln_predictor_alpha_tv",
        type=float,
        default=0.0,
        help="Weight of TV regularization term for AdaLN predictor training.",
    )
    parser.add_argument(
        "--adaln_predictor_tv_norm",
        type=str,
        choices=["l1", "l2"],
        default="l1",
        help="Norm used for TV regularization when --adaln_predictor_alpha_tv > 0.",
    )
    parser.add_argument(
        "--adaln_predictor_early_stop_patience",
        type=int,
        default=5,
        help=(
            "Stop predictor training early when validation loss does not improve for this many "
            "validation checks (<=0 disables)."
        ),
    )

    parser.add_argument(
        "--tta_enable",
        action="store_true",
        help="Enable per-image test-time optimization (resets parameters for each image).",
    )
    parser.add_argument("--tta_steps", type=int, default=40, help="TTO steps per image.")
    parser.add_argument("--tta_lr", type=float, default=5e-3, help="Learning rate for TTO.")
    parser.add_argument(
        "--tta_loss",
        type=str,
        choices=[
            "l1",
            "l2",
            "l1_l2",
            "liqe",
            "maniqa",
            "musiq",
            "nr_mix",
            "lpips_liqe",
            "aug_consistency",
            "lr_consistency",
        ],
        default="liqe",
        help="Optimization loss used during TTO.",
    )
    parser.add_argument(
        "--tta_lr_consistency_mode",
        type=str,
        choices=["l1", "perceptual"],
        default="l1",
        help=(
            "Distance used by --tta_loss=lr_consistency. "
            "'l1' compares bicubic downsampled SR and LR reference in pixel space; "
            "'perceptual' compares them with LPIPS."
        ),
    )
    parser.add_argument(
        "--tta_lr_consistency_scale",
        type=int,
        default=0,
        help=(
            "Downsampling factor used by --tta_loss=lr_consistency. "
            "Set to your SR scale factor (usually --upscale); <=0 uses --upscale."
        ),
    )
    parser.add_argument(
        "--tta_lr_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Optional LR consistency regularization weight added to the base TTO loss. "
            "Effective when > 0 and --tta_loss is not lr_consistency."
        ),
    )
    parser.add_argument(
        "--tta_aug_num_transforms",
        type=int,
        default=4,
        help=(
            "Number of random transforms to average in --tta_loss=aug_consistency. "
            "Each transform is applied consistently to both branches."
        ),
    )
    parser.add_argument(
        "--tta_aug_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for augmentation consistency regularization added to the base "
            "TTO loss. Effective when > 0 and --tta_loss is not aug_consistency."
        ),
    )
    parser.add_argument(
        "--tta_nr_mix_liqe_weight",
        type=float,
        default=0.6,
        help="LIQE weight used when --tta_loss=nr_mix.",
    )
    parser.add_argument(
        "--tta_nr_mix_maniqa_weight",
        type=float,
        default=0.3,
        help="MANIQA weight used when --tta_loss=nr_mix.",
    )
    parser.add_argument(
        "--tta_nr_mix_musiq_weight",
        type=float,
        default=0.1,
        help="MUSIQ weight used when --tta_loss=nr_mix.",
    )
    parser.add_argument(
        "--tta_nr_mix_normalize_weights",
        type=_str2bool,
        default=True,
        help="Normalize nr_mix weights so they sum to 1 before combining losses.",
    )
    parser.add_argument(
        "--tta_early_stop_patience",
        type=int,
        default=8,
        help="Early stop TTO when no loss improvement for this many steps (<=0 disables).",
    )
    parser.add_argument(
        "--tta_min_steps",
        type=int,
        default=1,
        help="Minimum number of TTO steps before early stopping can trigger.",
    )
    parser.add_argument(
        "--tta_min_delta",
        type=float,
        default=1e-4,
        help="Minimum loss improvement used by TTO early stopping.",
    )
    parser.add_argument(
        "--tta_weighted_lr",
        type=_str2bool,
        default=True,
        help="Scale TTO updates using phase-1 activation importance.",
    )
    parser.add_argument(
        "--tta_lr_scale_min",
        type=float,
        default=0.25,
        help="Minimum block scaling factor for TTO weighted updates.",
    )
    parser.add_argument(
        "--tta_lr_scale_max",
        type=float,
        default=2.0,
        help="Maximum block scaling factor for TTO weighted updates.",
    )
    parser.add_argument(
        "--tta_adaptive_k",
        type=_str2bool,
        default=True,
        help="Enable adaptive-k selection per block for TTO using phase-1 scores.",
    )
    parser.add_argument(
        "--tta_adaptive_k_mass",
        type=float,
        default=0.85,
        help="Cumulative score mass target used to choose adaptive-k.",
    )
    parser.add_argument(
        "--tta_adaptive_k_min",
        type=int,
        default=1,
        help="Minimum k per block for adaptive-k.",
    )
    parser.add_argument(
        "--tta_adaptive_k_max",
        type=int,
        default=0,
        help="Maximum k per block for adaptive-k (<=0 falls back to --topk).",
    )
    parser.add_argument(
        "--tta_std_percentile_mode",
        type=str,
        choices=["none", "per_stream", "global"],
        default="none",
        help=(
            "Alternative TTA channel selection based on percentile over phase-1 std rankings. "
            "'per_stream' keeps top percentile independently for hs/ehs; "
            "'global' ranks hs+ehs together and keeps top percentile globally."
        ),
    )
    parser.add_argument(
        "--tta_std_percentile",
        type=float,
        default=70.0,
        help=(
            "Percentile of channels to keep for --tta_std_percentile_mode. "
            "Example: 70 keeps top 70% ranked channels."
        ),
    )
    parser.add_argument(
        "--tta_zero_non_selected_channels",
        type=_str2bool,
        default=False,
        help=(
            "TTA modality: set all non-selected channels to zero in hooked blocks "
            "while continuing to optimize selected channels normally."
        ),
    )
    parser.add_argument(
        "--tta_max_images",
        type=int,
        default=0,
        help=(
            "Limit TTO/inference to a subset of N images for debugging. "
            "<=0 uses all inference images."
        ),
    )
    parser.add_argument(
        "--tta_shuffle_images",
        action="store_true",
        help="Randomly sample TTO subset when --tta_max_images > 0 (otherwise uses first N after sorting).",
    )

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--resource_log_every",
        type=int,
        default=250,
        help=(
            "Print runtime resource snapshots every N items in long loops "
            "(<=0 disables)."
        ),
    )
    parser.add_argument(
        "--eval_iqa",
        type=_str2bool,
        default=True,
        help="Compute full IQA evaluation over inference outputs and save global/per-image summaries.",
    )
    parser.add_argument(
        "--measure_flops",
        type=_str2bool,
        default=False,
        help=(
            "Measure FLOPs with torch.profiler for baseline/post-TTA inference and add them to timing reports. "
            "This adds profiling overhead."
        ),
    )
    parser.add_argument(
        "--measure_tta_optimization_flops",
        type=_str2bool,
        default=True,
        help=(
            "When --measure_flops is true, also profile FLOPs for the TTA optimization loop. "
            "Can be expensive for large --tta_steps."
        ),
    )
    parser.add_argument(
        "--reuse_existing_run",
        type=_str2bool,
        default=False,
        help="Reuse existing SR outputs in --output_dir/images and recompute reports/metrics without rerunning TTA.",
    )
    parser.add_argument(
        "--reuse_after_dir",
        type=str,
        default="",
        help="Directory with existing final SR images (defaults to --output_dir/images).",
    )

    return parser.parse_args()

def _adaln_modulate(
    tensor: torch.Tensor,
    channels: List[int],
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    if not channels:
        return tensor
    idx = torch.tensor(channels, device=tensor.device, dtype=torch.long)
    selected = torch.index_select(tensor, dim=2, index=idx)
    if gamma.numel() == 1:
        gamma = gamma.view(1, 1, 1)
    elif gamma.dim() == 1:
        gamma = gamma.view(1, 1, -1)
    elif gamma.dim() == 2 and gamma.shape[0] == selected.shape[0]:
        gamma = gamma.view(selected.shape[0], 1, -1)
    else:
        raise ValueError(
            f"Invalid gamma shape for AdaLN modulation: {tuple(gamma.shape)} with selected {tuple(selected.shape)}"
        )

    if beta.numel() == 1:
        beta = beta.view(1, 1, 1)
    elif beta.dim() == 1:
        beta = beta.view(1, 1, -1)
    elif beta.dim() == 2 and beta.shape[0] == selected.shape[0]:
        beta = beta.view(selected.shape[0], 1, -1)
    else:
        raise ValueError(
            f"Invalid beta shape for AdaLN modulation: {tuple(beta.shape)} with selected {tuple(selected.shape)}"
        )
    gamma = gamma.to(dtype=selected.dtype, device=selected.device)
    beta = beta.to(dtype=selected.dtype, device=selected.device)
    mod = selected * gamma + beta
    out = tensor.clone()
    out[:, :, idx] = mod
    return out


def _has_affine_like_params(params: Dict[str, Any], stream: str) -> bool:
    return f"gamma_{stream}" in params or f"beta_{stream}" in params


def _has_stream_modulation_params(params: Dict[str, Any], stream: str) -> bool:
    return _has_affine_like_params(params, stream) or f"res_w_{stream}" in params


def _zero_non_selected_channels(tensor: torch.Tensor, channels: List[int]) -> torch.Tensor:
    """Keep only selected channels and set all others to zero."""
    if tensor.dim() != 3:
        return tensor
    if not channels:
        return torch.zeros_like(tensor)
    idx = torch.tensor(channels, device=tensor.device, dtype=torch.long)
    out = torch.zeros_like(tensor)
    out[:, :, idx] = torch.index_select(tensor, dim=2, index=idx)
    return out

def _zero_selected_channels(tensor: torch.Tensor, channels: List[int]) -> torch.Tensor:
    if not channels:
        return tensor
    idx = torch.tensor(channels, device=tensor.device, dtype=torch.long)
    out = tensor.clone()
    out[:, :, idx] = 0
    return out

def _stream_choice_to_weights(choice: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if choice == "hs":
        return torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)
    if choice == "ehs":
        return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    return torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)

def _blend_stream_modulation(
    original: torch.Tensor,
    modulated: torch.Tensor,
    stream_weight: torch.Tensor,
) -> torch.Tensor:
    """Interpolate between original and modulated stream with scalar or per-batch weights."""
    if original.shape != modulated.shape:
        return modulated
    w = stream_weight.to(device=original.device, dtype=original.dtype)
    if w.numel() == 1:
        w = w.view(1, 1, 1)
    elif w.dim() == 1 and w.shape[0] == original.shape[0]:
        w = w.view(original.shape[0], 1, 1)
    else:
        raise ValueError(
            f"Invalid stream weight shape {tuple(stream_weight.shape)} for tensor batch {original.shape[0]}"
        )
    return original + w * (modulated - original)


def _adaln_residual_modulate(
    tensor: torch.Tensor,
    channels: List[int],
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
) -> torch.Tensor:
    if not channels:
        return tensor
    idx = torch.tensor(channels, device=tensor.device, dtype=torch.long)
    selected = torch.index_select(tensor, dim=2, index=idx)

    weight = proj_weight.to(dtype=selected.dtype, device=selected.device)
    bias = proj_bias.to(dtype=selected.dtype, device=selected.device)
    if weight.dim() == 1:
        proj = selected * weight.view(1, 1, -1)
    elif weight.dim() == 2 and weight.shape[0] == selected.shape[0] and weight.shape[1] == selected.shape[2]:
        proj = selected * weight.view(selected.shape[0], 1, -1)
    elif weight.dim() == 2:
        proj = torch.matmul(selected, weight.t())
    elif weight.dim() == 3 and weight.shape[0] == selected.shape[0]:
        proj = torch.einsum("bnc,bkc->bnk", selected, weight)
    else:
        raise ValueError(f"Invalid residual projection rank: {weight.dim()}")

    if bias.dim() == 2 and bias.shape[0] == selected.shape[0]:
        proj = proj + bias.view(selected.shape[0], 1, -1)
    else:
        proj = proj + bias.view(1, 1, -1)
    out = tensor.clone()
    out[:, :, idx] = selected + proj
    return out



class _BlockAdaLNFixedHook:
    def __init__(
        self,
        topk_info: Dict[str, Any],
        params: Dict[str, torch.Tensor],
        modulate_hidden: bool = True,
        modulate_encoder: bool = False,
        zero_hidden_channels: bool = False,
        zero_encoder_channels: bool = False,
        zero_hidden_non_selected: bool = False,
        zero_encoder_non_selected: bool = False,
        stream_weight_mode: str = "manual",
        stream_prior_ratio_low: float = 0.8,
        stream_prior_ratio_high: float = 1.2,
        stream_prior_eps: float = 1e-6,
    ):
        self.topk_info = topk_info
        self.params = params
        self.modulate_hidden = modulate_hidden
        self.modulate_encoder = modulate_encoder
        self.zero_hidden_channels = zero_hidden_channels
        self.zero_encoder_channels = zero_encoder_channels
        self.zero_hidden_non_selected = zero_hidden_non_selected
        self.zero_encoder_non_selected = zero_encoder_non_selected
        self.stream_weight_mode = stream_weight_mode
        self.stream_prior_ratio_low = stream_prior_ratio_low
        self.stream_prior_ratio_high = stream_prior_ratio_high
        self.stream_prior_eps = stream_prior_eps

    def __call__(self, module, inputs, output):
        encoder_hidden_states, hidden_states = output

        if self.zero_hidden_non_selected:
            hidden_states = _zero_non_selected_channels(hidden_states, self.topk_info["hs_topk_idx"])

        if self.zero_hidden_channels:
            hidden_states = _zero_selected_channels(hidden_states, self.topk_info["hs_topk_idx"])
        elif self.modulate_hidden:
            hs_weight = torch.tensor(1.0, device=hidden_states.device, dtype=hidden_states.dtype)
            if self.stream_weight_mode == "learned_gate_std_prior_offline":
                has_encoder_stream = bool(
                    encoder_hidden_states is not None
                    and self.modulate_encoder
                    and _has_stream_modulation_params(self.params, "ehs")
                )
                if has_encoder_stream:
                    stream_choice = _std_prior_stream_choice(
                        self.topk_info,
                        ratio_low=self.stream_prior_ratio_low,
                        ratio_high=self.stream_prior_ratio_high,
                        eps=self.stream_prior_eps,
                    )
                    hs_weight, _ = _stream_choice_to_weights(stream_choice, hidden_states.device)
                    hs_weight = hs_weight.to(dtype=hidden_states.dtype)
            if self.stream_weight_mode != "manual" and "stream_gate_weights" in self.params:
                has_encoder_stream = bool(
                    encoder_hidden_states is not None
                    and self.modulate_encoder
                    and _has_stream_modulation_params(self.params, "ehs")
                )
                if has_encoder_stream:
                    hs_weight = self.params["stream_gate_weights"][0].to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
            if _has_affine_like_params(self.params, "hs"):
                gamma = self.params.get(
                    "gamma_hs",
                    torch.tensor(1.0, device=hidden_states.device, dtype=hidden_states.dtype),
                )
                beta = self.params.get(
                    "beta_hs",
                    torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype),
                )
                hs_mod = _adaln_modulate(
                    hidden_states,
                    self.topk_info["hs_topk_idx"],
                    gamma,
                    beta,
                )
                hidden_states = _blend_stream_modulation(hidden_states, hs_mod, hs_weight)
            elif "res_w_hs" in self.params:
                hs_mod = _adaln_residual_modulate(
                    hidden_states,
                    self.topk_info["hs_topk_idx"],
                    self.params["res_w_hs"],
                    self.params["res_b_hs"],
                )
                hidden_states = _blend_stream_modulation(hidden_states, hs_mod, hs_weight)

        if self.zero_encoder_non_selected and encoder_hidden_states is not None:
            encoder_hidden_states = _zero_non_selected_channels(
                encoder_hidden_states,
                self.topk_info["ehs_topk_idx"],
            )

        if self.zero_encoder_channels and encoder_hidden_states is not None:
            encoder_hidden_states = _zero_selected_channels(encoder_hidden_states, self.topk_info["ehs_topk_idx"])
        elif self.modulate_encoder and encoder_hidden_states is not None:
            ehs_weight = torch.tensor(1.0, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            if self.stream_weight_mode == "learned_gate_std_prior_offline":
                has_hidden_stream = bool(
                    self.modulate_hidden and _has_stream_modulation_params(self.params, "hs")
                )
                if has_hidden_stream:
                    stream_choice = _std_prior_stream_choice(
                        self.topk_info,
                        ratio_low=self.stream_prior_ratio_low,
                        ratio_high=self.stream_prior_ratio_high,
                        eps=self.stream_prior_eps,
                    )
                    _, ehs_weight = _stream_choice_to_weights(stream_choice, encoder_hidden_states.device)
                    ehs_weight = ehs_weight.to(dtype=encoder_hidden_states.dtype)
            if self.stream_weight_mode != "manual" and "stream_gate_weights" in self.params:
                has_hidden_stream = bool(
                    self.modulate_hidden and _has_stream_modulation_params(self.params, "hs")
                )
                if has_hidden_stream:
                    ehs_weight = self.params["stream_gate_weights"][1].to(
                        device=encoder_hidden_states.device,
                        dtype=encoder_hidden_states.dtype,
                    )
            if _has_affine_like_params(self.params, "ehs"):
                gamma = self.params.get(
                    "gamma_ehs",
                    torch.tensor(1.0, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype),
                )
                beta = self.params.get(
                    "beta_ehs",
                    torch.tensor(0.0, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype),
                )
                ehs_mod = _adaln_modulate(
                    encoder_hidden_states,
                    self.topk_info["ehs_topk_idx"],
                    gamma,
                    beta,
                )
                encoder_hidden_states = _blend_stream_modulation(encoder_hidden_states, ehs_mod, ehs_weight)
            elif "res_w_ehs" in self.params:
                ehs_mod = _adaln_residual_modulate(
                    encoder_hidden_states,
                    self.topk_info["ehs_topk_idx"],
                    self.params["res_w_ehs"],
                    self.params["res_b_ehs"],
                )
                encoder_hidden_states = _blend_stream_modulation(encoder_hidden_states, ehs_mod, ehs_weight)

        return encoder_hidden_states, hidden_states



class AdaLNVaeConditionPredictor(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("AdaLNVaeConditionPredictor expects positive input/output dims.")
        hidden_dim = max(1, int(hidden_dim))
        num_layers = max(1, int(num_layers))
        dropout = float(max(0.0, dropout))

        layers: List[torch.nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, int(output_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _compute_stream_gate_weights(
    logits: torch.Tensor,
    stream_weight_mode: str,
    hard: bool = False,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return hidden/encoder stream weights and optional 3-state probabilities."""
    mode = "learned_gate_softmax2" if stream_weight_mode == "learned_gate" else stream_weight_mode
    l = logits.to(torch.float32)
    temp = max(1e-4, float(temperature))

    if mode == "learned_gate_softmax2":
        if l.dim() == 1:
            probs = torch.softmax(l[:2] / temp, dim=0)
            return probs[0], probs[1], None
        probs = torch.softmax(l[:, :2] / temp, dim=1)
        return probs[:, 0], probs[:, 1], None

    if mode == "learned_gate_hard_global_ste":
        if l.dim() == 1:
            probs = torch.softmax(l[:2] / temp, dim=0)
            if hard:
                idx = torch.argmax(probs)
                one_hot = torch.zeros_like(probs)
                one_hot[idx] = 1.0
                ste = one_hot - probs.detach() + probs
                return ste[0], ste[1], None
            return probs[0], probs[1], None
        probs = torch.softmax(l[:, :2] / temp, dim=1)
        if hard:
            idx = torch.argmax(probs, dim=1, keepdim=True)
            one_hot = torch.zeros_like(probs).scatter_(1, idx, 1.0)
            ste = one_hot - probs.detach() + probs
            return ste[:, 0], ste[:, 1], None
        return probs[:, 0], probs[:, 1], None

    if mode == "learned_gate_sigmoid2":
        if l.dim() == 1:
            gates = torch.sigmoid(l[:2])
            return gates[0], gates[1], None
        gates = torch.sigmoid(l[:, :2])
        return gates[:, 0], gates[:, 1], None

    if mode == "learned_gate_tristate":
        if l.dim() == 1:
            probs3 = torch.softmax(l[:3] / temp, dim=0)
            w_h = probs3[0] + probs3[2]
            w_e = probs3[1] + probs3[2]
            return w_h, w_e, probs3
        probs3 = torch.softmax(l[:, :3] / temp, dim=1)
        w_h = probs3[:, 0] + probs3[:, 2]
        w_e = probs3[:, 1] + probs3[:, 2]
        return w_h, w_e, probs3

    if l.dim() == 1:
        return torch.tensor(1.0, device=l.device), torch.tensor(1.0, device=l.device), None
    ones = torch.ones(l.shape[0], device=l.device)
    return ones, ones, None




class _BlockAdaLNHook:
    def __init__(
        self,
        block_idx: int,
        topk_info: Dict[str, Any],
        raw_params: Dict[str, torch.nn.Parameter],
        gamma_min: float,
        gamma_max: float,
        beta_min: float,
        beta_max: float,
        modulation_type: str = "residual",
        modulate_hidden: bool = True,
        modulate_encoder: bool = False,
        zero_hidden_channels: bool = False,
        zero_encoder_channels: bool = False,
        zero_hidden_non_selected: bool = False,
        zero_encoder_non_selected: bool = False,
        stream_weight_mode: str = "manual",
        global_stream_gate_logits: Optional[torch.nn.Parameter] = None,
        stream_gate_temperature: float = 1.0,
        stream_prior_ratio_low: float = 0.8,
        stream_prior_ratio_high: float = 1.2,
        stream_prior_eps: float = 1e-6,
        constrain_with_sigmoid: bool = True,
    ):
        self.block_idx = block_idx
        self.topk_info = topk_info
        self.raw_params = raw_params
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.modulation_type = modulation_type
        self.modulate_hidden = modulate_hidden
        self.modulate_encoder = modulate_encoder
        self.zero_hidden_channels = zero_hidden_channels
        self.zero_encoder_channels = zero_encoder_channels
        self.zero_hidden_non_selected = zero_hidden_non_selected
        self.zero_encoder_non_selected = zero_encoder_non_selected
        self.stream_weight_mode = stream_weight_mode
        self.global_stream_gate_logits = global_stream_gate_logits
        self.stream_gate_temperature = stream_gate_temperature
        self.stream_prior_ratio_low = stream_prior_ratio_low
        self.stream_prior_ratio_high = stream_prior_ratio_high
        self.stream_prior_eps = stream_prior_eps
        self.constrain_with_sigmoid = constrain_with_sigmoid

    def __call__(self, module, inputs, output):
        encoder_hidden_states, hidden_states = output

        if self.zero_hidden_non_selected:
            hidden_states = _zero_non_selected_channels(hidden_states, self.topk_info["hs_topk_idx"])

        if self.zero_hidden_channels:
            hidden_states = _zero_selected_channels(hidden_states, self.topk_info["hs_topk_idx"])
        elif self.modulate_hidden:
            hs_weight = torch.tensor(1.0, device=hidden_states.device, dtype=hidden_states.dtype)
            gate_logits = self.raw_params.get("stream_gate_logits", self.global_stream_gate_logits)
            if self.stream_weight_mode == "learned_gate_std_prior_offline":
                has_encoder_stream = bool(
                    encoder_hidden_states is not None
                    and self.modulate_encoder
                    and _has_stream_modulation_params(self.raw_params, "ehs")
                )
                if has_encoder_stream:
                    stream_choice = _std_prior_stream_choice(
                        self.topk_info,
                        ratio_low=self.stream_prior_ratio_low,
                        ratio_high=self.stream_prior_ratio_high,
                        eps=self.stream_prior_eps,
                    )
                    hs_weight, _ = _stream_choice_to_weights(stream_choice, hidden_states.device)
                    hs_weight = hs_weight.to(dtype=hidden_states.dtype)
            if self.stream_weight_mode != "manual" and gate_logits is not None:
                has_encoder_stream = bool(
                    encoder_hidden_states is not None
                    and self.modulate_encoder
                    and _has_stream_modulation_params(self.raw_params, "ehs")
                )
                if has_encoder_stream:
                    w_hs, _, _ = _compute_stream_gate_weights(
                        gate_logits,
                        self.stream_weight_mode,
                        hard=(self.stream_weight_mode == "learned_gate_hard_global_ste"),
                        temperature=self.stream_gate_temperature,
                    )
                    hs_weight = w_hs.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if self.modulation_type in {"affine", "scale", "shift"} and _has_affine_like_params(self.raw_params, "hs"):
                if "gamma_hs" in self.raw_params:
                    gamma = _project_param(
                        self.raw_params["gamma_hs"],
                        self.gamma_min,
                        self.gamma_max,
                        self.constrain_with_sigmoid,
                    )
                else:
                    gamma = torch.tensor(1.0, device=hidden_states.device, dtype=hidden_states.dtype)
                if "beta_hs" in self.raw_params:
                    beta = _project_param(
                        self.raw_params["beta_hs"],
                        self.beta_min,
                        self.beta_max,
                        self.constrain_with_sigmoid,
                    )
                else:
                    beta = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
                hs_mod = _adaln_modulate(hidden_states, self.topk_info["hs_topk_idx"], gamma, beta)
                hidden_states = _blend_stream_modulation(hidden_states, hs_mod, hs_weight)
            elif self.modulation_type == "residual" and "res_w_hs" in self.raw_params:
                hs_mod = _adaln_residual_modulate(
                    hidden_states,
                    self.topk_info["hs_topk_idx"],
                    self.raw_params["res_w_hs"],
                    self.raw_params["res_b_hs"],
                )
                hidden_states = _blend_stream_modulation(hidden_states, hs_mod, hs_weight)

        if self.zero_encoder_non_selected and encoder_hidden_states is not None:
            encoder_hidden_states = _zero_non_selected_channels(
                encoder_hidden_states,
                self.topk_info["ehs_topk_idx"],
            )

        if self.zero_encoder_channels and encoder_hidden_states is not None:
            encoder_hidden_states = _zero_selected_channels(encoder_hidden_states, self.topk_info["ehs_topk_idx"])
        elif self.modulate_encoder and encoder_hidden_states is not None:
            ehs_weight = torch.tensor(1.0, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            gate_logits = self.raw_params.get("stream_gate_logits", self.global_stream_gate_logits)
            if self.stream_weight_mode == "learned_gate_std_prior_offline":
                has_hidden_stream = bool(
                    self.modulate_hidden and _has_stream_modulation_params(self.raw_params, "hs")
                )
                if has_hidden_stream:
                    stream_choice = _std_prior_stream_choice(
                        self.topk_info,
                        ratio_low=self.stream_prior_ratio_low,
                        ratio_high=self.stream_prior_ratio_high,
                        eps=self.stream_prior_eps,
                    )
                    _, ehs_weight = _stream_choice_to_weights(stream_choice, encoder_hidden_states.device)
                    ehs_weight = ehs_weight.to(dtype=encoder_hidden_states.dtype)
            if self.stream_weight_mode != "manual" and gate_logits is not None:
                has_hidden_stream = bool(
                    self.modulate_hidden and _has_stream_modulation_params(self.raw_params, "hs")
                )
                if has_hidden_stream:
                    _, w_ehs, _ = _compute_stream_gate_weights(
                        gate_logits,
                        self.stream_weight_mode,
                        hard=(self.stream_weight_mode == "learned_gate_hard_global_ste"),
                        temperature=self.stream_gate_temperature,
                    )
                    ehs_weight = w_ehs.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            if self.modulation_type in {"affine", "scale", "shift"} and _has_affine_like_params(self.raw_params, "ehs"):
                if "gamma_ehs" in self.raw_params:
                    gamma = _project_param(
                        self.raw_params["gamma_ehs"],
                        self.gamma_min,
                        self.gamma_max,
                        self.constrain_with_sigmoid,
                    )
                else:
                    gamma = torch.tensor(1.0, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                if "beta_ehs" in self.raw_params:
                    beta = _project_param(
                        self.raw_params["beta_ehs"],
                        self.beta_min,
                        self.beta_max,
                        self.constrain_with_sigmoid,
                    )
                else:
                    beta = torch.tensor(0.0, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                ehs_mod = _adaln_modulate(
                    encoder_hidden_states, self.topk_info["ehs_topk_idx"], gamma, beta
                )
                encoder_hidden_states = _blend_stream_modulation(encoder_hidden_states, ehs_mod, ehs_weight)
            elif self.modulation_type == "residual" and "res_w_ehs" in self.raw_params:
                ehs_mod = _adaln_residual_modulate(
                    encoder_hidden_states,
                    self.topk_info["ehs_topk_idx"],
                    self.raw_params["res_w_ehs"],
                    self.raw_params["res_b_ehs"],
                )
                encoder_hidden_states = _blend_stream_modulation(encoder_hidden_states, ehs_mod, ehs_weight)

        return encoder_hidden_states, hidden_states



def _init_dinov2_cosine_metric(device: str) -> Optional[Any]:
    #cached_model_paths = [
    #    os.path.expanduser("~/.cache/torch/hub/checkpoints/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
    #    os.path.expanduser("~/.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
    #]
    
    #for cached_path in cached_model_paths:
    #    if os.path.isdir(cached_path):
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("facebook/dinov2-base", trust_remote_code=True, local_files_only=True)
        model.eval().to(device)
        image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        print(f"[dinov2] ✓ Successfully loaded DINOv2 from local")
        return _DINOv2CosineMetric(model, image_size=image_size, mean=mean, std=std, device=device)
    except Exception as exc:
        print(f"[warn] could not load DINOv2 from cached path local: {exc}")
    
    try:
        import timm
    except Exception as exc:
        print(f"[warn] timm unavailable for dinov2_cosine: {exc}")
        return None

    print("[dinov2] No cached model found, attempting to download via timm...")
    candidate_models = [
        "vit_base_patch14_dinov2.lvd142m",
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_reg4_dinov2.lvd142m",
    ]
    for model_name in candidate_models:
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
            model.eval().to(device)
            cfg = getattr(model, "default_cfg", {}) or {}
            input_size = cfg.get("input_size", (3, 224, 224))
            image_size = int(input_size[1]) if len(input_size) >= 2 else 224
            mean = list(cfg.get("mean", (0.485, 0.456, 0.406)))
            std = list(cfg.get("std", (0.229, 0.224, 0.225)))
            print(f"[dinov2] ✓ Successfully loaded DINOv2 '{model_name}' via timm")
            return _DINOv2CosineMetric(model, image_size=image_size, mean=mean, std=std, device=device)
        except Exception as exc:
            print(f"[warn] could not initialize DINOv2 model '{model_name}': {exc}")
    print("[warn] dinov2_cosine metric unavailable and will be skipped.")
    return None



def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_iqa_metrics(device: str, metric_names: List[str]) -> Dict[str, Any]:
    import pyiqa

    metrics: Dict[str, Any] = {}
    for name in metric_names:
        if name == "dinov2_cosine":
            metric = _init_dinov2_cosine_metric(device)
            if metric is not None:
                metrics[name] = metric
            continue

        if name == "ssim":
            # One SSIM convention for the whole codebase: Y-channel / YCbCr, the
            # same object the eval scripts build.  Plain RGB SSIM reads ~3.4
            # points lower on the same images, so mixing the two across tables
            # silently inflates or deflates every reported SSIM gap.
            metrics[name] = pyiqa.create_metric(
                "ssim", test_y_channel=True, color_space="ycbcr", device=device
            ).eval()
            continue

        candidates = [name]
        if name == "maniqa-pipal":
            candidates = ["maniqa-pipal", "maniqa"]

        loaded = None
        for c in candidates:
            try:
                loaded = pyiqa.create_metric(c, device=device)
                loaded.eval()
                break
            except Exception as exc:
                print(f"[warn] could not initialize IQA metric '{c}' (requested '{name}'): {exc}")
        if loaded is not None:
            metrics[name] = loaded
    return metrics

class _DINOv2CosineMetric:
    def __init__(self, model: Any, image_size: int, mean: List[float], std: List[float], device: str):
        self.model = model
        self.image_size = int(image_size)
        self.device = device
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _features(self, x_bchw01: torch.Tensor) -> torch.Tensor:
        x = x_bchw01.to(self.device, dtype=torch.float32)
        x = torch.nn.functional.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = (x - self.mean) / self.std
        feat = self.model(x)
        
        # Handle transformers BaseModelOutputWithPooling (has .last_hidden_state, .pooler_output)
        if hasattr(feat, 'last_hidden_state'):
            feat = feat.last_hidden_state
        elif hasattr(feat, 'pooler_output'):
            feat = feat.pooler_output
        elif isinstance(feat, (tuple, list)):
            feat = feat[0]
        
        # feat should now be a tensor; extract image-level feature if spatial dims exist
        if feat.dim() > 2:
            dims = tuple(range(2, feat.dim()))
            feat = feat.mean(dim=dims)
        return torch.nn.functional.normalize(feat.float(), dim=1)

    @torch.no_grad()
    def __call__(self, pred_bchw01: torch.Tensor, ref_bchw01: torch.Tensor) -> torch.Tensor:
        pred_f = self._features(pred_bchw01)
        ref_f = self._features(ref_bchw01)
        return (pred_f * ref_f).sum(dim=1).mean()


def _list_images_in_folder(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )

def _collect_pairs(input_dir: str, gt_dir: Optional[str]) -> List[Dict[str, Optional[str]]]:
    if os.path.isdir(input_dir):
        lr_paths = _list_images_in_folder(input_dir)
    else:
        lr_paths = [input_dir]

    pairs: List[Dict[str, Optional[str]]] = []
    for lr_path in lr_paths:
        name = os.path.basename(lr_path)
        gt_path = None
        if gt_dir:
            candidate = os.path.join(gt_dir, name)
            if os.path.exists(candidate):
                gt_path = candidate
        pairs.append({"lr": lr_path, "gt": gt_path})
    return pairs


def load_model(args, weight_dtype: torch.dtype):
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
    )

    if args.is_use_tile:
        _init_tiled_vae(vae, encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "proj", "linear", "proj_out"],
    )
    transformer.add_adapter(transformer_lora_config)
    transformer.enable_adapters()

    vae_target_modules = [
        "encoder.conv_in",
        "encoder.down_blocks.0.resnets.0.conv1",
        "encoder.down_blocks.0.resnets.0.conv2",
        "encoder.down_blocks.0.resnets.1.conv1",
        "encoder.down_blocks.0.resnets.1.conv2",
        "encoder.down_blocks.0.downsamplers.0.conv",
        "encoder.down_blocks.1.resnets.0.conv1",
        "encoder.down_blocks.1.resnets.0.conv2",
        "encoder.down_blocks.1.resnets.0.conv_shortcut",
        "encoder.down_blocks.1.resnets.1.conv1",
        "encoder.down_blocks.1.resnets.1.conv2",
        "encoder.down_blocks.1.downsamplers.0.conv",
        "encoder.down_blocks.2.resnets.0.conv1",
        "encoder.down_blocks.2.resnets.0.conv2",
        "encoder.down_blocks.2.resnets.0.conv_shortcut",
        "encoder.down_blocks.2.resnets.1.conv1",
        "encoder.down_blocks.2.resnets.1.conv2",
        "encoder.down_blocks.2.downsamplers.0.conv",
        "encoder.down_blocks.3.resnets.0.conv1",
        "encoder.down_blocks.3.resnets.0.conv2",
        "encoder.down_blocks.3.resnets.1.conv1",
        "encoder.down_blocks.3.resnets.1.conv2",
        "encoder.mid_block.attentions.0.to_q",
        "encoder.mid_block.attentions.0.to_k",
        "encoder.mid_block.attentions.0.to_v",
        "encoder.mid_block.attentions.0.to_out.0",
        "encoder.mid_block.resnets.0.conv1",
        "encoder.mid_block.resnets.0.conv2",
        "encoder.mid_block.resnets.1.conv1",
        "encoder.mid_block.resnets.1.conv2",
        "encoder.conv_out",
        "quant_conv",
    ]
    vae_lora_config = LoraConfig(
        r=args.rank_vae,
        lora_alpha=args.rank_vae,
        init_lora_weights="gaussian",
        target_modules=vae_target_modules,
    )
    vae.add_adapter(vae_lora_config)
    vae.enable_adapters()

    vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="vae.safetensors")
    transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir, weight_name="transformer.safetensors")

    load_lora_state_dict(vae_lora_state_dict, vae)
    load_lora_state_dict(transformer_lora_state_dict, transformer)

    vae = vae.to(args.device, dtype=weight_dtype)
    transformer = transformer.to(args.device, dtype=weight_dtype)

    prompt_embeds = torch.load(os.path.join(args.embedding_dir, "prompt_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)
    pooled_prompt_embeds = torch.load(os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)

    return transformer, vae, prompt_embeds, pooled_prompt_embeds


def _normalize_topk_map(raw_topk: Any) -> Optional[Dict[int, Dict[str, Any]]]:
    if not isinstance(raw_topk, dict):
        return None
    out: Dict[int, Dict[str, Any]] = OrderedDict()
    for key, value in raw_topk.items():
        if not isinstance(value, dict):
            continue
        try:
            block_idx = int(key)
        except Exception:
            continue
        out[block_idx] = {
            "hs_topk_idx": value.get("hs_topk_idx", []),
            "hs_topk_val": value.get("hs_topk_val", []),
            "ehs_topk_idx": value.get("ehs_topk_idx"),
            "ehs_topk_val": value.get("ehs_topk_val"),
            "hs_score_full": value.get("hs_score_full"),
            "ehs_score_full": value.get("ehs_score_full"),
            "hs_std_full": value.get("hs_std_full"),
            "ehs_std_full": value.get("ehs_std_full"),
        }
    return out if out else None



def _load_cached_topk(cache_path: str) -> Tuple[Optional[Dict[int, Dict[str, Any]]], Optional[Dict[int, Dict[str, Any]]]]:
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "phase1_topk" in payload:
        topk_info = _normalize_topk_map(payload.get("phase1_topk"))
        bottomk_info = _normalize_topk_map(payload.get("phase1_bottomk"))
    else:
        topk_info = _normalize_topk_map(payload)
        bottomk_info = None

    if topk_info is None:
        raise RuntimeError(
            f"Could not parse cached topk info from '{cache_path}'. "
            "Expected either a phase report JSON with 'phase1_topk' or a raw topk map."
        )

    return topk_info, bottomk_info
        
def _select_phase1_pairs(
    pairs: List[Dict[str, Optional[str]]],
    max_samples: int,
    seed: Optional[int] = None,
    shuffle: bool = False,
) -> List[Dict[str, Optional[str]]]:
    """Select a subset of optimization pairs for phase-1 top-k aggregation."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(pairs):
        return pairs

    if not shuffle:
        return pairs[:max_samples]

    indices = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = set(indices[:max_samples])
    return [pair for idx, pair in enumerate(pairs) if idx in chosen]


def _load_rgb_image(path: str) -> Image.Image:
    """Load an image as RGB and close the file descriptor immediately."""
    with Image.open(path) as img:
        return img.convert("RGB").copy()

def _compute_target_size(
    ori_width: int,
    ori_height: int,
    upscale: int,
    process_size: int,
) -> Tuple[int, int, bool]:
    resize_flag = False
    if ori_width < process_size // upscale or ori_height < process_size // upscale:
        scale = (process_size // upscale) / min(ori_width, ori_height)
        new_width, new_height = int(scale * ori_width), int(scale * ori_height)
        resize_flag = True
    else:
        new_width, new_height = ori_width, ori_height

    new_width, new_height = upscale * new_width, upscale * new_height
    if new_width % 8 or new_height % 8:
        resize_flag = True
        new_width = new_width - new_width % 8
        new_height = new_height - new_height % 8
    return new_width, new_height, resize_flag


def _prepare_tensor(lr: Image.Image, size: Tuple[int, int], device: str, weight_dtype: torch.dtype) -> torch.Tensor:
    pixel_values = transforms.ToTensor()(lr).unsqueeze(0)
    pixel_values = torch.nn.functional.interpolate(pixel_values, size=size, mode="bicubic", align_corners=False)
    pixel_values = pixel_values * 2 - 1
    return pixel_values.to(device, dtype=weight_dtype).clamp(-1, 1)


def _phase1_loss_requires_gt(args) -> bool:
    if args.phase1_grad_loss in {"l1", "l2", "l1_l2", "lpips"}:
        return True
    if args.phase1_grad_loss == "hybrid" and args.phase1_hybrid_recon_weight > 0:
        return True
    return False


def _register_capture_hooks(
    transformer: SD3Transformer2DModel,
    captures: List[Dict[str, Any]],
    capture_grad: bool = False,
) -> List[Any]:
    handles = []

    for block_idx, block in enumerate(transformer.transformer_blocks):
        def _hook(module, inputs, output, idx=block_idx):
            encoder_hidden_states, hidden_states = output
            if capture_grad:
                if hidden_states is not None and hidden_states.requires_grad:
                    hidden_states.retain_grad()
                if encoder_hidden_states is not None and encoder_hidden_states.requires_grad:
                    encoder_hidden_states.retain_grad()
            captures.append(
                {
                    "block_index": idx,
                    "hidden_states": hidden_states,
                    "encoder_hidden_states": encoder_hidden_states,
                }
            )
            return output

        handles.append(block.register_forward_hook(_hook))

    return handles

def _encode_input(vae: AutoencoderKL, pixel_values: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
    return model_input.to(device=pixel_values.device, dtype=weight_dtype)

def _gaussian_weights(tile_width: int, tile_height: int, nbatches: int, device: str, channels: int) -> torch.Tensor:
    from numpy import exp, pi, sqrt
    import numpy as np

    var = 0.01
    midpoint_x = (tile_width - 1) / 2
    x_probs = [exp(-(x - midpoint_x) * (x - midpoint_x) / (tile_width * tile_width) / (2 * var)) / sqrt(2 * pi * var)
               for x in range(tile_width)]
    midpoint_y = tile_height / 2
    y_probs = [exp(-(y - midpoint_y) * (y - midpoint_y) / (tile_height * tile_height) / (2 * var)) / sqrt(2 * pi * var)
               for y in range(tile_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=device), (nbatches, channels, 1, 1))



def _tile_sample(
    lq_latent: torch.Tensor,
    lq: torch.Tensor,
    transformer: SD3Transformer2DModel,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    weight_dtype: torch.dtype,
    args,
    use_no_grad: bool = True,
) -> torch.Tensor:
    context = torch.no_grad() if use_no_grad else torch.enable_grad()
    with context:
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (args.latent_tiled_size, args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            model_pred = transformer(
                hidden_states=lq_latent,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        else:
            tile_size = min(tile_size, min(h, w))
            tile_weights = _gaussian_weights(tile_size, tile_size, 1, args.device, transformer.config.in_channels)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

                    model_out = transformer(
                        hidden_states=input_tile.to(args.device, dtype=weight_dtype),
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    noise_preds.append(model_out)

            noise_pred = torch.zeros(lq_latent.shape, device=args.device)
            contributors = torch.zeros(lq_latent.shape, device=args.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += (
                        noise_preds[row * grid_cols + col] * tile_weights
                    )
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            noise_pred /= contributors
            model_pred = noise_pred

    return model_pred.to(args.device, dtype=weight_dtype)



def _run_sr(
    vae: AutoencoderKL,
    transformer: SD3Transformer2DModel,
    pixel_values: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    weight_dtype: torch.dtype,
    args,
    use_no_grad: bool = True,
) -> torch.Tensor:
    context = torch.no_grad() if use_no_grad else torch.enable_grad()
    with context:
        model_input = _encode_input(vae, pixel_values, weight_dtype)
        if args.is_use_tile:
            model_pred = _tile_sample(
                model_input,
                pixel_values,
                transformer,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                weight_dtype,
                args,
                use_no_grad=use_no_grad,
            )
        else:
            model_pred = transformer(
                hidden_states=model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        latent_stu = model_input - model_pred
        image = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1, 1)
    return image


def _phase1_ig_interpolate(
    pixel_values: torch.Tensor,
    baseline: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Linear interpolation between baseline and input for IG."""
    return baseline + alpha * (pixel_values - baseline)


def _sr_to_bchw01(sr_tensor: torch.Tensor) -> torch.Tensor:
    sr_01 = (sr_tensor.clamp(-1, 1) + 1) / 2
    if sr_01.dim() == 3:
        sr_01 = sr_01.unsqueeze(0)
    elif sr_01.dim() != 4:
        raise ValueError(f"Expected SR tensor with 3D/4D shape, got {tuple(sr_01.shape)}")
    return sr_01.float()


def _compute_grad_loss(
    sr_tensor: torch.Tensor,
    gt_tensor: Optional[torch.Tensor],
    loss_type: str,
    liqe_metric: Optional[Any] = None,
    maniqa_metric: Optional[Any] = None,
    musiq_metric: Optional[Any] = None,
    lpips_metric: Optional[Any] = None,
    lpips_weight: float = 1.0,
    liqe_weight: float = 1.0,
) -> torch.Tensor:
    if loss_type == "liqe":
        if liqe_metric is None:
            raise ValueError("liqe_metric must be provided when using LIQE loss.")
        sr_01 = (sr_tensor.clamp(-1, 1) + 1) / 2
        if sr_01.dim() == 3:
            sr_01 = sr_01.unsqueeze(0)
        elif sr_01.dim() != 4:
            raise ValueError(f"LIQE expects 3D/4D tensor, got shape {tuple(sr_01.shape)}")
        return -liqe_metric(sr_01.float())

    if loss_type == "maniqa":
        if maniqa_metric is None:
            raise ValueError("maniqa_metric must be provided when using MANIQA loss.")
        sr_01 = _sr_to_bchw01(sr_tensor)
        return -maniqa_metric(sr_01.float())

    if loss_type == "musiq":
        if musiq_metric is None:
            raise ValueError("musiq_metric must be provided when using MUSIQ loss.")
        sr_01 = _sr_to_bchw01(sr_tensor)
        return -musiq_metric(sr_01.float())

    if gt_tensor is None:
        raise ValueError("gt_tensor is required for non-NR losses.")

    if loss_type == "l2":
        return torch.mean((sr_tensor - gt_tensor) ** 2)
    if loss_type == "lpips":
        if lpips_metric is None:
            raise ValueError("lpips_metric must be provided when using LPIPS loss.")
        sr_01 = _sr_to_bchw01(sr_tensor)
        gt_01 = _sr_to_bchw01(gt_tensor)
        return lpips_metric(sr_01, gt_01)
    if loss_type == "l1_l2":
        l1 = torch.mean(torch.abs(sr_tensor - gt_tensor))
        l2 = torch.mean((sr_tensor - gt_tensor) ** 2)
        return 0.5 * l1 + 0.5 * l2
    if loss_type == "lpips_liqe":
        if liqe_metric is None:
            raise ValueError("liqe_metric must be provided when using LPIPS+LIQE loss.")
        if lpips_metric is None:
            raise ValueError("lpips_metric must be provided when using LPIPS+LIQE loss.")
        if gt_tensor is None:
            raise ValueError("gt_tensor is required for LPIPS+LIQE loss.")

        sr_01 = _sr_to_bchw01(sr_tensor)
        gt_01 = _sr_to_bchw01(gt_tensor)
        loss_liqe = -liqe_metric(sr_01)
        loss_lpips = lpips_metric(sr_01, gt_01)
        return lpips_weight * loss_lpips + liqe_weight * loss_liqe
    return torch.mean(torch.abs(sr_tensor - gt_tensor))



def _compute_phase1_grad_loss(
    sr_tensor: torch.Tensor,
    gt_tensor: Optional[torch.Tensor],
    args,
    nr_metric: Optional[Any] = None,
    lpips_metric: Optional[Any] = None,
) -> torch.Tensor:
    if args.phase1_grad_loss in {"l1", "l2", "l1_l2", "lpips"}:
        return _compute_grad_loss(
            sr_tensor,
            gt_tensor,
            loss_type=args.phase1_grad_loss,
            lpips_metric=lpips_metric,
        )

    if args.phase1_grad_loss in PHASE1_NR_METRICS:
        if nr_metric is None:
            raise RuntimeError("Phase-1 NR metric is not initialized.")
        sr_01 = _sr_to_bchw01(sr_tensor)
        return -nr_metric(sr_01)

    if args.phase1_grad_loss == "hybrid":
        if nr_metric is None:
            raise RuntimeError("Phase-1 hybrid NR metric is not initialized.")

        total = torch.tensor(0.0, device=sr_tensor.device, dtype=sr_tensor.dtype)
        if args.phase1_hybrid_recon_weight > 0:
            recon = _compute_grad_loss(
                sr_tensor,
                gt_tensor,
                loss_type=args.phase1_hybrid_recon_loss,
                lpips_metric=lpips_metric,
            )
            total = total + float(args.phase1_hybrid_recon_weight) * recon

        if args.phase1_hybrid_nr_weight > 0:
            sr_01 = _sr_to_bchw01(sr_tensor)
            nr = -nr_metric(sr_01)
            total = total + float(args.phase1_hybrid_nr_weight) * nr

        return total

    raise ValueError(f"Unknown phase1_grad_loss: {args.phase1_grad_loss}")

def _stabilize_grad_tensor(
    grad: Optional[torch.Tensor],
    loss: Optional[torch.Tensor],
    loss_norm_mode: str,
    clip_norm: float,
    eps: float,
) -> Optional[torch.Tensor]:
    if grad is None:
        return None

    g = grad.to(torch.float32)
    if loss is not None and loss_norm_mode != "none":
        scale = torch.abs(loss.detach().to(torch.float32))
        if loss_norm_mode == "loss_sqrt":
            scale = torch.sqrt(torch.clamp(scale, min=eps))
        scale = torch.clamp(scale, min=eps)
        if torch.isfinite(scale).item():
            g = g / scale

    if clip_norm > 0:
        g_norm = torch.norm(g)
        if torch.isfinite(g_norm).item() and g_norm.item() > clip_norm:
            g = g * (clip_norm / (g_norm + eps))

    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


def _channel_gradxact_vector(
    tensor: torch.Tensor,
    grad: Optional[torch.Tensor],
    mode: str = "normalized_grad_act",
    eps: float = 1e-6,
) -> torch.Tensor:
    if grad is None:
        raise RuntimeError("Expected gradient tensor for gradxact importance, but got None.")
    t = tensor.to(torch.float32)
    g = grad.to(torch.float32)

    if mode == "grad_only":
        score = torch.abs(g)
    elif mode == "normalized_grad_act":
        # Normalize per-channel activations across tokens to reduce scale bias.
        t_mean = t.mean(dim=1, keepdim=True)
        t_std = t.std(dim=1, keepdim=True)
        t_norm = (t - t_mean) / torch.clamp(t_std, min=eps)
        score = torch.abs(g * t_norm)
    else:
        # Legacy behavior.
        score = torch.abs(g * t)

    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    score = score.mean(dim=1)
    if score.dim() > 1:
        score = score.mean(dim=0)
    return score

import resource
def _bytes_to_mib(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _log_resource_snapshot(phase: str, step: int, total: int) -> None:
    """Print a compact runtime resource snapshot for long-running loops."""
    fd_count = -1
    try:
        fd_count = len(os.listdir("/proc/self/fd"))
    except Exception:
        pass

    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = float(rss_kib) / 1024.0

    gpu_msg = "gpu=cpu"
    if torch.cuda.is_available():
        alloc = _bytes_to_mib(torch.cuda.memory_allocated())
        reserved = _bytes_to_mib(torch.cuda.memory_reserved())
        max_alloc = _bytes_to_mib(torch.cuda.max_memory_allocated())
        gpu_msg = f"gpu_alloc={alloc:.1f}MiB gpu_reserved={reserved:.1f}MiB gpu_peak={max_alloc:.1f}MiB"

    print(
        f"[resource] phase={phase} step={step}/{total} "
        f"rss_peak={rss_mib:.1f}MiB fd={fd_count} {gpu_msg}"
    )

def _channel_std_vector(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.to(torch.float32)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    std = t.std(dim=1)
    if std.dim() > 1:
        std = std.mean(dim=0)
    return std


def _channel_mean_abs_vector(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.to(torch.float32)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    if t.dim() <= 1:
        return torch.abs(t)
    reduce_dims = tuple(range(t.dim() - 1))
    return torch.abs(t).mean(dim=reduce_dims)


def _channel_kurtosis_vector(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    t = tensor.to(torch.float32)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    mu = t.mean(dim=1, keepdim=True)
    centered = t - mu
    m2 = torch.mean(centered * centered, dim=1)
    m4 = torch.mean(centered * centered * centered * centered, dim=1)
    kurt = m4 / torch.clamp(m2 * m2, min=eps)
    excess = torch.clamp(kurt - 3.0, min=0.0)
    if excess.dim() > 1:
        excess = excess.mean(dim=0)
    return torch.nan_to_num(excess, nan=0.0, posinf=0.0, neginf=0.0)


def _channel_snr_proxy_vector(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    t = tensor.to(torch.float32)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    mean_abs = torch.abs(t.mean(dim=1))
    std = torch.clamp(t.std(dim=1), min=eps)
    snr = mean_abs / std
    if snr.dim() > 1:
        snr = snr.mean(dim=0)
    return torch.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)


def _aggregate_channel_samples(
    samples: List[torch.Tensor],
    mode: str = "mean",
    trim_fraction: float = 0.1,
) -> torch.Tensor:
    if not samples:
        raise RuntimeError("No channel samples were provided for aggregation.")
    stack = torch.stack([s.to(torch.float32) for s in samples], dim=0)
    stack = torch.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "mean":
        return stack.mean(dim=0)
    if mode == "median":
        return stack.median(dim=0).values
    if mode == "trimmed_mean":
        trim_fraction = min(0.49, max(0.0, float(trim_fraction)))
        n = stack.size(0)
        k = int(float(trim_fraction) * n)
        if k <= 0 or (2 * k) >= n:
            return stack.mean(dim=0)
        sorted_stack, _ = torch.sort(stack, dim=0)
        return sorted_stack[k:n - k].mean(dim=0)
    raise ValueError(f"Unknown phase1 aggregation mode: {mode}")

def _cross_image_spread(samples: List[torch.Tensor], mode: str = "std", eps: float = 1e-6) -> torch.Tensor:
    if not samples:
        raise RuntimeError("No cross-image samples were provided.")
    stack = torch.stack([s.to(torch.float32) for s in samples], dim=0)
    stack = torch.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "std":
        return stack.std(dim=0)
    if mode == "mad":
        med = stack.median(dim=0).values
        return (stack - med).abs().median(dim=0).values
    if mode == "iqr":
        q1 = torch.quantile(stack, 0.25, dim=0)
        q3 = torch.quantile(stack, 0.75, dim=0)
        return torch.clamp(q3 - q1, min=eps)
    raise ValueError(f"Unknown phase1 cross-image spread mode: {mode}")


def _robust_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    v = torch.nan_to_num(x.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    med = torch.median(v)
    mad = torch.median(torch.abs(v - med))
    scale = torch.clamp(1.4826 * mad, min=eps)
    return (v - med) / scale

def _select_k_from_scores(scores: torch.Tensor, topk: int, selection_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    k = min(int(topk), int(scores.numel()))
    if k <= 0:
        return (
            torch.empty(0, device=scores.device, dtype=scores.dtype),
            torch.empty(0, device=scores.device, dtype=torch.long),
        )

    if selection_mode == "random":
        idx = torch.randperm(scores.numel(), device=scores.device)[:k]
        return scores[idx], idx

    if selection_mode == "bottomk":
        return torch.topk(scores, k=k, largest=False)

    if selection_mode == "middlek":
        # Select channels from the middle of the score ranking.
        _, sorted_idx = torch.sort(scores, descending=True)
        start = max(0, (scores.numel() - k) // 2)
        idx = sorted_idx[start:start + k]
        return scores[idx], idx

    if selection_mode == "topk":
        return torch.topk(scores, k=k, largest=True)

    raise ValueError(f"Unknown selection_mode: {selection_mode}")




def _aggregate_topk_from_dataset(
    pairs: List[Dict[str, Optional[str]]],
    vae: AutoencoderKL,
    transformer: SD3Transformer2DModel,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    weight_dtype: torch.dtype,
    args,
    phase1_nr_metric: Optional[Any] = None,
    phase1_lpips_metric: Optional[Any] = None,
    include_score_vectors: bool = False,
) -> Dict[int, Dict[str, Any]]:
    sums: Dict[int, Dict[str, Any]] = OrderedDict()
    importance_mode = getattr(args, "phase1_importance_mode", "mean_abs")
    use_grad_importance = importance_mode == "gradxact"

    for idx, pair in enumerate(tqdm(pairs, desc="phase1-topk"), start=1):
        try:
            lr = _load_rgb_image(pair["lr"])
        except Exception as exc:
            raise RuntimeError(f"Failed to load input image '{pair['lr']}': {exc}") from exc
        ori_width, ori_height = lr.size
        new_width, new_height, _ = _compute_target_size(ori_width, ori_height, args.upscale, args.process_size)
        pixel_values = _prepare_tensor(lr, (new_height, new_width), args.device, weight_dtype)

        gt_needed = use_grad_importance and _phase1_loss_requires_gt(args)
        if gt_needed and pair.get("gt") is None:
            raise RuntimeError(
                "phase1 grad loss requires matching GT images for each phase-1 sample."
            )

        gt_tensor = None
        if gt_needed:
            try:
                gt = _load_rgb_image(pair["gt"])
            except Exception as exc:
                raise RuntimeError(f"Failed to load GT image '{pair['gt']}': {exc}") from exc

            gt_h, gt_w = int(ori_height * args.upscale), int(ori_width * args.upscale)
            gt_tensor = _prepare_tensor(gt, (gt_h, gt_w), args.device, weight_dtype).squeeze(0)

        if use_grad_importance and args.phase1_gradxact_mode == "integrated_gradients":
            ig_steps = max(1, int(args.phase1_ig_steps))
            eps = float(args.phase1_grad_eps)

            if args.phase1_ig_baseline == "zero":
                ig_baseline = torch.zeros_like(pixel_values)
            else:
                raise ValueError(f"Unknown phase1_ig_baseline: {args.phase1_ig_baseline}")

            # Capture baseline activations at alpha=0 (no backward needed).
            baseline_captures: List[Dict[str, Any]] = []
            baseline_handles = _register_capture_hooks(transformer, baseline_captures, capture_grad=False)
            try:
                _ = _run_sr(
                    vae,
                    transformer,
                    ig_baseline,
                    timesteps,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    weight_dtype,
                    args,
                    use_no_grad=True,
                )
            finally:
                for h in baseline_handles:
                    h.remove()

            hs_baseline: Dict[int, torch.Tensor] = {}
            ehs_baseline: Dict[int, Optional[torch.Tensor]] = {}
            for act in baseline_captures:
                block_idx = act["block_index"]
                hs_baseline[block_idx] = act["hidden_states"].detach().to(torch.float32)
                ehs = act.get("encoder_hidden_states")
                ehs_baseline[block_idx] = ehs.detach().to(torch.float32) if ehs is not None else None

            hs_grad_sums: Dict[int, torch.Tensor] = {}
            ehs_grad_sums: Dict[int, Optional[torch.Tensor]] = {}
            hs_last: Dict[int, torch.Tensor] = {}
            ehs_last: Dict[int, Optional[torch.Tensor]] = {}
            grad_norm_diag: Dict[int, float] = {}

            for step_idx in range(1, ig_steps + 1):
                alpha = float(step_idx) / float(ig_steps)
                ig_input = _phase1_ig_interpolate(pixel_values, ig_baseline, alpha).detach().requires_grad_(True)

                step_captures: List[Dict[str, Any]] = []
                step_handles = _register_capture_hooks(transformer, step_captures, capture_grad=True)
                loss_for_stabilization: Optional[torch.Tensor] = None
                try:
                    transformer.zero_grad(set_to_none=True)
                    vae.zero_grad(set_to_none=True)
                    sr = _run_sr(
                        vae,
                        transformer,
                        ig_input,
                        timesteps,
                        prompt_embeds,
                        pooled_prompt_embeds,
                        weight_dtype,
                        args,
                        use_no_grad=False,
                    )
                    loss = _compute_phase1_grad_loss(
                        sr,
                        gt_tensor,
                        args,
                        nr_metric=phase1_nr_metric,
                        lpips_metric=phase1_lpips_metric,
                    )
                    loss.backward()
                    loss_for_stabilization = loss
                finally:
                    for h in step_handles:
                        h.remove()

                for act in step_captures:
                    block_idx = act["block_index"]
                    hs_last[block_idx] = act["hidden_states"].detach().to(torch.float32)
                    hs_grad = _stabilize_grad_tensor(
                        act["hidden_states"].grad,
                        loss=loss_for_stabilization,
                        loss_norm_mode=args.phase1_loss_norm_mode,
                        clip_norm=float(args.phase1_grad_clip_norm),
                        eps=eps,
                    )
                    if hs_grad is not None:
                        hs_grad = hs_grad.detach().to(torch.float32)
                        if block_idx not in hs_grad_sums:
                            hs_grad_sums[block_idx] = torch.zeros_like(hs_grad)
                        hs_grad_sums[block_idx] = hs_grad_sums[block_idx] + hs_grad

                    ehs = act.get("encoder_hidden_states")
                    if ehs is None:
                        ehs_last[block_idx] = None
                        if block_idx not in ehs_grad_sums:
                            ehs_grad_sums[block_idx] = None
                    else:
                        ehs_last[block_idx] = ehs.detach().to(torch.float32)
                        ehs_grad = _stabilize_grad_tensor(
                            ehs.grad,
                            loss=loss_for_stabilization,
                            loss_norm_mode=args.phase1_loss_norm_mode,
                            clip_norm=float(args.phase1_grad_clip_norm),
                            eps=eps,
                        )
                        if ehs_grad is not None:
                            ehs_grad = ehs_grad.detach().to(torch.float32)
                            if block_idx not in ehs_grad_sums or ehs_grad_sums[block_idx] is None:
                                ehs_grad_sums[block_idx] = torch.zeros_like(ehs_grad)
                            ehs_grad_sums[block_idx] = ehs_grad_sums[block_idx] + ehs_grad

            for block_idx in hs_last.keys():
                hs_delta = hs_last[block_idx] - hs_baseline[block_idx]
                hs_avg_grad = hs_grad_sums[block_idx] / float(ig_steps)
                hs_score = _channel_gradxact_vector(
                    hs_delta,
                    hs_avg_grad,
                    mode="abs_grad_act",
                    eps=eps,
                ).detach().to(device="cpu", dtype=torch.float32)
                grad_norm_diag[block_idx] = float(torch.norm(hs_avg_grad).detach().cpu().item())

                ehs_score = None
                ehs_base = ehs_baseline.get(block_idx)
                ehs_end = ehs_last.get(block_idx)
                ehs_grad_sum = ehs_grad_sums.get(block_idx)
                if ehs_base is not None and ehs_end is not None and ehs_grad_sum is not None:
                    ehs_delta = ehs_end - ehs_base
                    ehs_avg_grad = ehs_grad_sum / float(ig_steps)
                    ehs_score = _channel_gradxact_vector(
                        ehs_delta,
                        ehs_avg_grad,
                        mode="abs_grad_act",
                        eps=eps,
                    ).detach().to(device="cpu", dtype=torch.float32)

                if block_idx not in sums:
                    sums[block_idx] = {
                        "hs_sum": torch.zeros_like(hs_score),
                        "ehs_sum": torch.zeros_like(ehs_score) if ehs_score is not None else None,
                        "count": 0,
                    }

                sums[block_idx]["hs_sum"] = sums[block_idx]["hs_sum"] + hs_score
                if ehs_score is not None:
                    if sums[block_idx]["ehs_sum"] is None:
                        sums[block_idx]["ehs_sum"] = torch.zeros_like(ehs_score)
                    sums[block_idx]["ehs_sum"] = sums[block_idx]["ehs_sum"] + ehs_score
                sums[block_idx]["count"] += 1

            if args.phase1_grad_diag_every > 0 and (idx % args.phase1_grad_diag_every == 0) and grad_norm_diag:
                first_block = min(grad_norm_diag.keys())
                last_block = max(grad_norm_diag.keys())
                print(
                    "[phase1-grad] "
                    f"sample={idx}/{len(pairs)} mode=integrated_gradients "
                    f"ig_steps={ig_steps} loss_norm={args.phase1_loss_norm_mode} "
                    f"clip={args.phase1_grad_clip_norm} "
                    f"block{first_block}_norm={grad_norm_diag[first_block]:.3e} "
                    f"block{last_block}_norm={grad_norm_diag[last_block]:.3e}"
                )

            if args.resource_log_every > 0 and (idx % args.resource_log_every == 0):
                _log_resource_snapshot("phase1-topk", idx, len(pairs))
            continue

        block_captures: List[Dict[str, Any]] = []
        loss_for_stabilization: Optional[torch.Tensor] = None
        handles = _register_capture_hooks(transformer, block_captures, capture_grad=use_grad_importance)
        try:
            if use_grad_importance:
                transformer.zero_grad(set_to_none=True)
                vae.zero_grad(set_to_none=True)
                pixel_values = pixel_values.detach().requires_grad_(True)
                sr = _run_sr(
                    vae,
                    transformer,
                    pixel_values,
                    timesteps,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    weight_dtype,
                    args,
                    use_no_grad=False,
                )
                loss = _compute_phase1_grad_loss(
                    sr,
                    gt_tensor,
                    args,
                    nr_metric=phase1_nr_metric,
                    lpips_metric=phase1_lpips_metric,
                )
                loss.backward()
                loss_for_stabilization = loss
            else:
                _ = _run_sr(
                    vae,
                    transformer,
                    pixel_values,
                    timesteps,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    weight_dtype,
                    args,
                    use_no_grad=True,
                )
        finally:
            for h in handles:
                h.remove()

        grad_norm_diag: Dict[int, float] = {}
        for act in block_captures:
            block_idx = act["block_index"]
            if use_grad_importance:
                hs_grad = _stabilize_grad_tensor(
                    act["hidden_states"].grad,
                    loss=loss_for_stabilization,
                    loss_norm_mode=args.phase1_loss_norm_mode,
                    clip_norm=float(args.phase1_grad_clip_norm),
                    eps=float(args.phase1_grad_eps),
                )
                hs_score = _channel_gradxact_vector(
                    act["hidden_states"],
                    hs_grad,
                    mode=args.phase1_gradxact_mode,
                    eps=float(args.phase1_grad_eps),
                ).detach().to(device="cpu", dtype=torch.float32)
                if hs_grad is not None:
                    grad_norm_diag[block_idx] = float(torch.norm(hs_grad).detach().cpu().item())
            else:
                hs_std_score = _channel_std_vector(act["hidden_states"]).detach().to(device="cpu", dtype=torch.float32)
                hs_mean_abs_score = _channel_mean_abs_vector(act["hidden_states"]).detach().to(device="cpu", dtype=torch.float32)
                hs_kurt_score = _channel_kurtosis_vector(
                    act["hidden_states"], eps=float(args.phase1_grad_eps)
                ).detach().to(device="cpu", dtype=torch.float32)
                hs_snr_score = _channel_snr_proxy_vector(
                    act["hidden_states"], eps=float(args.phase1_grad_eps)
                ).detach().to(device="cpu", dtype=torch.float32)
                hs_img_mean_abs = hs_mean_abs_score

                if importance_mode == "std":
                    hs_score = hs_std_score
                elif importance_mode == "mean_abs":
                    hs_score = hs_mean_abs_score
                elif importance_mode == "kurtosis":
                    hs_score = hs_kurt_score
                elif importance_mode == "snr_proxy":
                    hs_score = hs_snr_score
                elif importance_mode == "fused_stats":
                    hs_score = hs_mean_abs_score
                else:
                    raise ValueError(f"Unknown phase1_importance_mode: {importance_mode}")
            ehs = act.get("encoder_hidden_states")
            if ehs is not None:
                if use_grad_importance:
                    ehs_grad = _stabilize_grad_tensor(
                        ehs.grad,
                        loss=loss_for_stabilization,
                        loss_norm_mode=args.phase1_loss_norm_mode,
                        clip_norm=float(args.phase1_grad_clip_norm),
                        eps=float(args.phase1_grad_eps),
                    )
                    ehs_score = _channel_gradxact_vector(
                        ehs,
                        ehs_grad,
                        mode=args.phase1_gradxact_mode,
                        eps=float(args.phase1_grad_eps),
                    ).detach().to(device="cpu", dtype=torch.float32)
                else:
                    ehs_std_score = _channel_std_vector(ehs).detach().to(device="cpu", dtype=torch.float32)
                    ehs_mean_abs_score = _channel_mean_abs_vector(ehs).detach().to(device="cpu", dtype=torch.float32)
                    ehs_kurt_score = _channel_kurtosis_vector(
                        ehs, eps=float(args.phase1_grad_eps)
                    ).detach().to(device="cpu", dtype=torch.float32)
                    ehs_snr_score = _channel_snr_proxy_vector(
                        ehs, eps=float(args.phase1_grad_eps)
                    ).detach().to(device="cpu", dtype=torch.float32)
                    ehs_img_mean_abs = ehs_mean_abs_score

                    if importance_mode == "std":
                        ehs_score = ehs_std_score
                    elif importance_mode == "mean_abs":
                        ehs_score = ehs_mean_abs_score
                    elif importance_mode == "kurtosis":
                        ehs_score = ehs_kurt_score
                    elif importance_mode == "snr_proxy":
                        ehs_score = ehs_snr_score
                    elif importance_mode == "fused_stats":
                        ehs_score = ehs_mean_abs_score
                    else:
                        raise ValueError(f"Unknown phase1_importance_mode: {importance_mode}")
            else:
                ehs_score = None

            if block_idx not in sums:
                block_stats: Dict[str, Any] = {
                    "hs_sum": torch.zeros_like(hs_score),
                    "ehs_sum": torch.zeros_like(ehs_score) if ehs_score is not None else None,
                    "count": 0,
                }
                if not use_grad_importance:
                    block_stats["hs_scores"] = []
                    block_stats["hs_std_scores"] = []
                    block_stats["hs_mean_abs_scores"] = []
                    block_stats["hs_kurtosis_scores"] = []
                    block_stats["hs_snr_scores"] = []
                    block_stats["hs_cross_image_values"] = []
                    block_stats["ehs_scores"] = [] if ehs_score is not None else None
                    block_stats["ehs_std_scores"] = [] if ehs_score is not None else None
                    block_stats["ehs_mean_abs_scores"] = [] if ehs_score is not None else None
                    block_stats["ehs_kurtosis_scores"] = [] if ehs_score is not None else None
                    block_stats["ehs_snr_scores"] = [] if ehs_score is not None else None
                    block_stats["ehs_cross_image_values"] = [] if ehs_score is not None else None
                sums[block_idx] = block_stats

            sums[block_idx]["hs_sum"] = sums[block_idx]["hs_sum"] + hs_score
            if ehs_score is not None:
                if sums[block_idx]["ehs_sum"] is None:
                    sums[block_idx]["ehs_sum"] = torch.zeros_like(ehs_score)
                sums[block_idx]["ehs_sum"] = sums[block_idx]["ehs_sum"] + ehs_score
            sums[block_idx]["count"] += 1

            if not use_grad_importance:
                sums[block_idx]["hs_scores"].append(hs_score)
                sums[block_idx]["hs_std_scores"].append(hs_std_score)
                sums[block_idx]["hs_mean_abs_scores"].append(hs_mean_abs_score)
                sums[block_idx]["hs_kurtosis_scores"].append(hs_kurt_score)
                sums[block_idx]["hs_snr_scores"].append(hs_snr_score)
                sums[block_idx]["hs_cross_image_values"].append(hs_img_mean_abs)

                if ehs_score is not None:
                    if sums[block_idx]["ehs_scores"] is None:
                        sums[block_idx]["ehs_scores"] = []
                        sums[block_idx]["ehs_std_scores"] = []
                        sums[block_idx]["ehs_mean_abs_scores"] = []
                        sums[block_idx]["ehs_kurtosis_scores"] = []
                        sums[block_idx]["ehs_snr_scores"] = []
                        sums[block_idx]["ehs_cross_image_values"] = []
                    sums[block_idx]["ehs_scores"].append(ehs_score)
                    sums[block_idx]["ehs_std_scores"].append(ehs_std_score)
                    sums[block_idx]["ehs_mean_abs_scores"].append(ehs_mean_abs_score)
                    sums[block_idx]["ehs_kurtosis_scores"].append(ehs_kurt_score)
                    sums[block_idx]["ehs_snr_scores"].append(ehs_snr_score)
                    sums[block_idx]["ehs_cross_image_values"].append(ehs_img_mean_abs)

        if use_grad_importance and args.phase1_grad_diag_every > 0 and (idx % args.phase1_grad_diag_every == 0):
            if grad_norm_diag:
                first_block = min(grad_norm_diag.keys())
                last_block = max(grad_norm_diag.keys())
                print(
                    "[phase1-grad] "
                    f"sample={idx}/{len(pairs)} mode={args.phase1_gradxact_mode} "
                    f"loss_norm={args.phase1_loss_norm_mode} clip={args.phase1_grad_clip_norm} "
                    f"block{first_block}_norm={grad_norm_diag[first_block]:.3e} "
                    f"block{last_block}_norm={grad_norm_diag[last_block]:.3e}"
                )

        if args.resource_log_every > 0 and (idx % args.resource_log_every == 0):
            _log_resource_snapshot("phase1-topk", idx, len(pairs))

    info: Dict[int, Dict[str, Any]] = OrderedDict()
    selection_mode = args.selection_mode
    topk = args.topk

    def _select_phase1_score(stats: Dict[str, Any], prefix: str) -> Optional[torch.Tensor]:
        sum_key = f"{prefix}_sum"
        if sum_key not in stats or stats[sum_key] is None:
            return None

        if use_grad_importance:
            return stats[sum_key] / float(stats["count"])

        if importance_mode != "fused_stats":
            samples = stats.get(f"{prefix}_scores")
            if samples:
                return _aggregate_channel_samples(
                    samples,
                    mode=args.phase1_channel_agg,
                    trim_fraction=float(args.phase1_trim_fraction),
                )
            return stats[sum_key] / float(stats["count"])

        std_score = _aggregate_channel_samples(
            stats.get(f"{prefix}_std_scores", []),
            mode=args.phase1_channel_agg,
            trim_fraction=float(args.phase1_trim_fraction),
        )
        mean_abs_score = _aggregate_channel_samples(
            stats.get(f"{prefix}_mean_abs_scores", []),
            mode=args.phase1_channel_agg,
            trim_fraction=float(args.phase1_trim_fraction),
        )
        kurtosis_score = _aggregate_channel_samples(
            stats.get(f"{prefix}_kurtosis_scores", []),
            mode=args.phase1_channel_agg,
            trim_fraction=float(args.phase1_trim_fraction),
        )
        snr_score = _aggregate_channel_samples(
            stats.get(f"{prefix}_snr_scores", []),
            mode=args.phase1_channel_agg,
            trim_fraction=float(args.phase1_trim_fraction),
        )
        cross_image_score = _cross_image_spread(
            stats.get(f"{prefix}_cross_image_values", []),
            mode=args.phase1_cross_image_metric,
            eps=float(args.phase1_grad_eps),
        )

        z_mean_abs = _robust_zscore(mean_abs_score, eps=float(args.phase1_grad_eps))
        z_kurtosis = _robust_zscore(kurtosis_score, eps=float(args.phase1_grad_eps))
        z_snr = _robust_zscore(snr_score, eps=float(args.phase1_grad_eps))
        z_cross = _robust_zscore(cross_image_score, eps=float(args.phase1_grad_eps))
        z_std = _robust_zscore(std_score, eps=float(args.phase1_grad_eps))

        fused = (
            float(args.phase1_fused_weight_mean_abs) * z_mean_abs
            + float(args.phase1_fused_weight_kurtosis) * z_kurtosis
            + float(args.phase1_fused_weight_snr) * z_snr
            + float(args.phase1_fused_weight_cross_image) * z_cross
            - float(args.phase1_fused_weight_std_penalty) * z_std
        )
        return torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

    for block_idx, stats in sums.items():
        hs_mean = _select_phase1_score(stats, "hs")
        if hs_mean is None:
            continue
        hs_std_full: Optional[torch.Tensor] = None
        if not use_grad_importance:
            hs_std_samples = stats.get("hs_std_scores")
            if hs_std_samples:
                hs_std_full = _aggregate_channel_samples(
                    hs_std_samples,
                    mode=args.phase1_channel_agg,
                    trim_fraction=float(args.phase1_trim_fraction),
                )
        hs_vals, hs_idx = _select_k_from_scores(hs_mean, topk=topk, selection_mode=selection_mode)

        ehs_idx_list = None
        ehs_val_list = None
        ehs_mean = _select_phase1_score(stats, "ehs")
        ehs_std_full: Optional[torch.Tensor] = None
        if ehs_mean is not None:
            if not use_grad_importance:
                ehs_std_samples = stats.get("ehs_std_scores")
                if ehs_std_samples:
                    ehs_std_full = _aggregate_channel_samples(
                        ehs_std_samples,
                        mode=args.phase1_channel_agg,
                        trim_fraction=float(args.phase1_trim_fraction),
                    )
            ehs_vals, ehs_idx = _select_k_from_scores(ehs_mean, topk=topk, selection_mode=selection_mode)
            ehs_idx_list = ehs_idx.detach().cpu().tolist()
            ehs_val_list = ehs_vals.detach().cpu().tolist()

        info[block_idx] = {
            "hs_topk_idx": hs_idx.detach().cpu().tolist(),
            "hs_topk_val": hs_vals.detach().cpu().tolist(),
            "ehs_topk_idx": ehs_idx_list,
            "ehs_topk_val": ehs_val_list,
        }
        if include_score_vectors:
            info[block_idx]["hs_score_full"] = hs_mean.detach().cpu().tolist()
            info[block_idx]["ehs_score_full"] = (
                ehs_mean.detach().cpu().tolist() if ehs_mean is not None else None
            )
            info[block_idx]["hs_std_full"] = (
                hs_std_full.detach().cpu().tolist() if hs_std_full is not None else None
            )
            info[block_idx]["ehs_std_full"] = (
                ehs_std_full.detach().cpu().tolist() if ehs_std_full is not None else None
            )

    return info


def _block_in_scope(block_idx: int, num_blocks: int, layer_scope: str) -> bool:
    if layer_scope == "all":
        return True

    early_end = max(1, (num_blocks + 2) // 3)
    mid_end = max(early_end, (2 * num_blocks + 2) // 3)

    if layer_scope == "early":
        return block_idx < early_end
    if layer_scope == "mid":
        return early_end <= block_idx < mid_end
    if layer_scope == "late":
        return block_idx >= mid_end
    raise ValueError(f"Unknown layer_scope: {layer_scope}")



def _filter_topk_by_layer_scope(
    topk_map: Optional[Dict[int, Dict[str, Any]]],
    num_blocks: int,
    layer_scope: str,
) -> Optional[Dict[int, Dict[str, Any]]]:
    if topk_map is None:
        return None
    if layer_scope == "all":
        return topk_map

    out: Dict[int, Dict[str, Any]] = OrderedDict()
    for block_idx, payload in topk_map.items():
        if _block_in_scope(int(block_idx), num_blocks, layer_scope):
            out[int(block_idx)] = payload
    return out


def _prepare_optimization_samples(
    pairs: List[Dict[str, Optional[str]]],
    args,
    weight_dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for idx, pair in enumerate(tqdm(pairs, desc="phase2-prepare"), start=1):
        samples.append({
            "name": os.path.basename(pair["lr"]),
            "lr_path": pair["lr"],
            "gt_path": pair.get("gt"),
        })

        if args.resource_log_every > 0 and (idx % args.resource_log_every == 0):
            _log_resource_snapshot("phase2-prepare", idx, len(pairs))
    return samples

def _prepare_metric_samples(
    pairs: List[Dict[str, Optional[str]]],
    args,
    weight_dtype: torch.dtype,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if max_samples is not None:
        pairs = pairs[: max(0, int(max_samples))]

    for idx, pair in enumerate(tqdm(pairs, desc="phase2-prepare-metrics"), start=1):
        samples.append({
            "name": os.path.basename(pair["lr"]),
            "lr_path": pair["lr"],
            "gt_path": pair.get("gt"),
        })

        if args.resource_log_every > 0 and (idx % args.resource_log_every == 0):
            _log_resource_snapshot("phase2-prepare-metrics", idx, len(pairs))
    return samples
    
def _materialize_opt_sample(
    sample_spec: Dict[str, Any],
    args,
    weight_dtype: torch.dtype,
) -> Dict[str, Any]:
    lr_path = sample_spec["lr_path"]
    try:
        lr = _load_rgb_image(lr_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load optimization input image '{lr_path}': {exc}") from exc

    ori_width, ori_height = lr.size
    new_width, new_height, _ = _compute_target_size(ori_width, ori_height, args.upscale, args.process_size)
    pixel_values = _prepare_tensor(lr, (new_height, new_width), args.device, weight_dtype)

    gt_tensor = None
    gt_path = sample_spec.get("gt_path")
    if gt_path is not None:
        try:
            gt = _load_rgb_image(gt_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load optimization GT image '{gt_path}': {exc}") from exc
        gt_h, gt_w = int(ori_height * args.upscale), int(ori_width * args.upscale)
        gt_tensor = _prepare_tensor(gt, (gt_h, gt_w), args.device, weight_dtype).squeeze(0)

    if args.adaln_loss != "liqe" and gt_tensor is None:
        raise RuntimeError("adaln optimization with non-LIQE loss requires matching GT images.")

    return {
        "name": sample_spec.get("name", os.path.basename(lr_path)),
        "pixel_values": pixel_values,
        "gt_tensor": gt_tensor,
    }

def _extract_vae_condition_features(
    vae: AutoencoderKL,
    pixel_values: torch.Tensor,
    weight_dtype: torch.dtype,
    feature_type: str = "mean_std",
) -> torch.Tensor:
    with torch.no_grad():
        posterior = vae.encode(pixel_values)
        latent = posterior.latent_dist.mode() * vae.config.scaling_factor
        latent = latent.to(device=pixel_values.device, dtype=weight_dtype)

    latent_f = latent.to(torch.float32)
    pooled_mean = latent_f.mean(dim=(-2, -1))
    if feature_type == "mean":
        features = pooled_mean
    elif feature_type == "mean_std":
        pooled_std = latent_f.std(dim=(-2, -1))
        features = torch.cat([pooled_mean, pooled_std], dim=1)
    else:
        raise ValueError(f"Unknown adaln_predictor_feature_type: {feature_type}")
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _build_adaln_raw_params(
    topk_per_block: Dict[int, Dict[str, Any]],
    per_channel: bool,
    device: torch.device,
    modulation_type: str = "residual",
    modulate_hidden: bool = True,
    modulate_encoder: bool = False,
    stream_weight_mode: str = "manual",
    constrain_with_sigmoid: bool = True,
) -> Dict[int, Dict[str, torch.nn.Parameter]]:
    raw_params: Dict[int, Dict[str, torch.nn.Parameter]] = {}
    if modulation_type not in {"affine", "scale", "shift", "residual"}:
        raise ValueError(f"Unknown adaln modulation_type: {modulation_type}")
    if stream_weight_mode not in {
        "manual",
        "learned_gate",
        "learned_gate_softmax2",
        "learned_gate_sigmoid2",
        "learned_gate_tristate",
        "learned_gate_hard_global_ste",
        "learned_gate_std_prior_offline",
        "liqe_best3_forward",
    }:
        raise ValueError(f"Unknown stream_weight_mode: {stream_weight_mode}")

    gamma_init = 0.0 if constrain_with_sigmoid else 1.0
    beta_init = 0.0
    for block_idx, info in topk_per_block.items():
        params: Dict[str, torch.nn.Parameter] = {}
        hs_idx = list(info.get("hs_topk_idx") or [])
        ehs_idx = list(info.get("ehs_topk_idx") or [])
        has_hs = bool(modulate_hidden and hs_idx)
        has_ehs = bool(modulate_encoder and ehs_idx)

        if has_hs:
            k = len(hs_idx)
            if modulation_type in {"affine", "scale", "shift"}:
                shape = (k,) if per_channel else (1,)
                if modulation_type in {"affine", "scale"}:
                    params["gamma_hs"] = torch.nn.Parameter(torch.full(shape, gamma_init, device=device))
                if modulation_type in {"affine", "shift"}:
                    params["beta_hs"] = torch.nn.Parameter(torch.full(shape, beta_init, device=device))
            else:
                if per_channel:
                    params["res_w_hs"] = torch.nn.Parameter(torch.zeros(k, device=device))
                else:
                    params["res_w_hs"] = torch.nn.Parameter(torch.zeros((k, k), device=device))
                params["res_b_hs"] = torch.nn.Parameter(torch.zeros(k, device=device))
        if has_ehs:
            k = len(ehs_idx)
            if modulation_type in {"affine", "scale", "shift"}:
                shape = (k,) if per_channel else (1,)
                if modulation_type in {"affine", "scale"}:
                    params["gamma_ehs"] = torch.nn.Parameter(torch.full(shape, gamma_init, device=device))
                if modulation_type in {"affine", "shift"}:
                    params["beta_ehs"] = torch.nn.Parameter(torch.full(shape, beta_init, device=device))
            else:
                if per_channel:
                    params["res_w_ehs"] = torch.nn.Parameter(torch.zeros(k, device=device))
                else:
                    params["res_w_ehs"] = torch.nn.Parameter(torch.zeros((k, k), device=device))
                params["res_b_ehs"] = torch.nn.Parameter(torch.zeros(k, device=device))
        if stream_weight_mode in {
            "learned_gate",
            "learned_gate_softmax2",
            "learned_gate_sigmoid2",
            "learned_gate_tristate",
        } and has_hs and has_ehs:
            gate_dim = 3 if stream_weight_mode == "learned_gate_tristate" else 2
            params["stream_gate_logits"] = torch.nn.Parameter(torch.zeros(gate_dim, device=device))
        if params:
            raw_params[block_idx] = params
    return raw_params




def _adaln_param_shape_spec(
    topk_per_block: Dict[int, Dict[str, Any]],
    per_channel: bool,
    device: torch.device,
    modulation_type: str = "residual",
    modulate_hidden: bool = True,
    modulate_encoder: bool = False,
    stream_weight_mode: str = "manual",
    constrain_with_sigmoid: bool = True,
) -> Dict[int, Dict[str, Tuple[int, ...]]]:
    raw_params = _build_adaln_raw_params(
        topk_per_block=topk_per_block,
        per_channel=per_channel,
        device=device,
        modulation_type=modulation_type,
        modulate_hidden=modulate_hidden,
        modulate_encoder=modulate_encoder,
        stream_weight_mode=stream_weight_mode,
        constrain_with_sigmoid=constrain_with_sigmoid,
    )
    return {
        int(block_idx): {
            name: tuple(param.shape)
            for name, param in block_params.items()
        }
        for block_idx, block_params in raw_params.items()
    }


def _adaln_param_numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return int(n)


def _flatten_adaln_param_spec(
    spec: Dict[int, Dict[str, Tuple[int, ...]]]
) -> Tuple[List[Tuple[int, str, Tuple[int, ...], slice]], int]:
    layout: List[Tuple[int, str, Tuple[int, ...], slice]] = []
    offset = 0
    for block_idx in sorted(spec.keys()):
        block_spec = spec[block_idx]
        for name in sorted(block_spec.keys()):
            shape = tuple(block_spec[name])
            count = _adaln_param_numel(shape)
            layout.append((int(block_idx), name, shape, slice(offset, offset + count)))
            offset += count
    return layout, offset

def _pil_to_tensor_minus1_1(image: Image.Image, device: str, weight_dtype: torch.dtype) -> torch.Tensor:
    t = transforms.ToTensor()(image)
    t = t * 2 - 1
    return t.to(device=device, dtype=weight_dtype).clamp(-1, 1)


def _tensor_to_pil(image_chw: torch.Tensor) -> Image.Image:
    image_01 = (image_chw.clamp(-1, 1) + 1) / 2
    # NumPy does not support bfloat16 tensors; convert to float32 on CPU first.
    image_hwc = image_01.permute(1, 2, 0).detach().to(device="cpu", dtype=torch.float32).numpy()
    image_hwc = (image_hwc * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(image_hwc)


def _postprocess_sr_like_classic(
    sr_tensor: torch.Tensor,
    lr_image: Image.Image,
    ori_width: int,
    ori_height: int,
    upscale: int,
    resize_flag: bool,
    align_method: str,
    device: str,
    weight_dtype: torch.dtype,
) -> Tuple[Image.Image, torch.Tensor]:
    out_w, out_h = int(ori_width * upscale), int(ori_height * upscale)
    sr_image = _tensor_to_pil(sr_tensor)
    if resize_flag:
        sr_image = sr_image.resize((out_w, out_h), Image.BICUBIC)

    if align_method == "adain":
        sr_image = adain_color_fix(target=sr_image, source=lr_image)
    elif align_method == "wavelet":
        lr_scale = lr_image.resize((out_w, out_h), Image.BICUBIC)
        sr_image = wavelet_color_fix(target=sr_image, source=lr_scale)

    sr_eval_tensor = _pil_to_tensor_minus1_1(sr_image, device=device, weight_dtype=weight_dtype)
    return sr_image, sr_eval_tensor


def _materialize_metric_sample(
    sample_spec: Dict[str, Any],
    args,
    weight_dtype: torch.dtype,
) -> Dict[str, Any]:
    lr_path = sample_spec["lr_path"]
    try:
        lr = _load_rgb_image(lr_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load metric input image '{lr_path}': {exc}") from exc

    ori_width, ori_height = lr.size
    new_width, new_height, resize_flag = _compute_target_size(ori_width, ori_height, args.upscale, args.process_size)
    pixel_values = _prepare_tensor(lr, (new_height, new_width), args.device, weight_dtype)

    gt_tensor = None
    gt_image = None
    gt_path = sample_spec.get("gt_path")
    if gt_path is not None:
        try:
            gt = _load_rgb_image(gt_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load metric GT image '{gt_path}': {exc}") from exc
        gt_image = gt
        gt_h, gt_w = int(ori_height * args.upscale), int(ori_width * args.upscale)
        gt_tensor = _prepare_tensor(gt, (gt_h, gt_w), args.device, weight_dtype).squeeze(0)

    return {
        "name": sample_spec.get("name", os.path.basename(lr_path)),
        "pixel_values": pixel_values,
        "gt_tensor": gt_tensor,
        "gt_image": gt_image,
        "lr_image": lr,
        "ori_width": ori_width,
        "ori_height": ori_height,
        "resize_flag": resize_flag,
    }

def _std_prior_stream_choice(
    topk_info: Dict[str, Any],
    ratio_high: float = 1.2,
    ratio_low: float = 0.8,
    eps: float = 1e-6,
) -> str:
    """Choose hs/ehs/both from phase-1 std statistics."""
    hs_vals = list(topk_info.get("hs_topk_val") or [])
    ehs_vals = list(topk_info.get("ehs_topk_val") or [])
    hs_mean = (sum(hs_vals) / len(hs_vals)) if hs_vals else 0.0
    ehs_mean = (sum(ehs_vals) / len(ehs_vals)) if ehs_vals else 0.0
    ratio = float(ehs_mean) / max(float(hs_mean), float(max(eps, 1e-12)))
    if ratio > float(ratio_high):
        return "ehs"
    if ratio < float(ratio_low):
        return "hs"
    return "both"


def _collect_std_prior_choices(
    topk_per_block: Dict[int, Dict[str, Any]],
    ratio_low: float,
    ratio_high: float,
    eps: float,
) -> Dict[int, str]:
    choices: Dict[int, str] = OrderedDict()
    for block_idx, info in topk_per_block.items():
        choices[int(block_idx)] = _std_prior_stream_choice(
            info,
            ratio_low=ratio_low,
            ratio_high=ratio_high,
            eps=eps,
        )
    return choices



def _log_std_prior_choices(
    prefix: str,
    topk_per_block: Dict[int, Dict[str, Any]],
    ratio_low: float,
    ratio_high: float,
    eps: float,
) -> None:
    choices = _collect_std_prior_choices(
        topk_per_block,
        ratio_low=ratio_low,
        ratio_high=ratio_high,
        eps=eps,
    )
    if not choices:
        print(f"[stream][std_prior] {prefix}: no blocks available for stream choice.")
        return
    counts = {"hs": 0, "ehs": 0, "both": 0}
    for c in choices.values():
        if c in counts:
            counts[c] += 1
    details = ", ".join([f"b{b}:{c}" for b, c in choices.items()])
    print(
        f"[stream][std_prior] {prefix}: hs={counts['hs']} ehs={counts['ehs']} both={counts['both']} "
        f"(ratio_low={ratio_low:.4f}, ratio_high={ratio_high:.4f}, eps={eps:.2e})"
    )
    print(f"[stream][std_prior] {prefix} per-block: {details}")




def _register_fixed_adaln_hooks(
    transformer: SD3Transformer2DModel,
    topk_per_block: Dict[int, Dict[str, Any]],
    adaln_snapshot: Dict[int, Dict[str, List[float]]],
    device: torch.device,
    modulate_hidden: bool = True,
    modulate_encoder: bool = False,
    zero_hidden_channels: bool = False,
    zero_encoder_channels: bool = False,
    zero_hidden_non_selected: bool = False,
    zero_encoder_non_selected: bool = False,
    stream_weight_mode: str = "manual",
    stream_prior_ratio_low: float = 0.8,
    stream_prior_ratio_high: float = 1.2,
    stream_prior_eps: float = 1e-6,
) -> List[Any]:
    handles = []
    if stream_weight_mode == "learned_gate_std_prior_offline":
        _log_std_prior_choices(
            prefix="fixed-hooks",
            topk_per_block=topk_per_block,
            ratio_low=float(stream_prior_ratio_low),
            ratio_high=float(stream_prior_ratio_high),
            eps=float(stream_prior_eps),
        )
    for block_idx, block in enumerate(transformer.transformer_blocks):
        if block_idx not in topk_per_block:
            continue
        snap = adaln_snapshot.get(block_idx, {})
        params: Dict[str, torch.Tensor] = {}
        if "gamma_hs" in snap:
            params["gamma_hs"] = torch.tensor(snap["gamma_hs"], device=device, dtype=torch.float32)
        if "beta_hs" in snap:
            params["beta_hs"] = torch.tensor(snap["beta_hs"], device=device, dtype=torch.float32)
        if "gamma_ehs" in snap:
            params["gamma_ehs"] = torch.tensor(snap["gamma_ehs"], device=device, dtype=torch.float32)
        if "beta_ehs" in snap:
            params["beta_ehs"] = torch.tensor(snap["beta_ehs"], device=device, dtype=torch.float32)
        if "res_w_hs" in snap and "res_b_hs" in snap:
            params["res_w_hs"] = torch.tensor(snap["res_w_hs"], device=device, dtype=torch.float32)
            params["res_b_hs"] = torch.tensor(snap["res_b_hs"], device=device, dtype=torch.float32)
        if "res_w_ehs" in snap and "res_b_ehs" in snap:
            params["res_w_ehs"] = torch.tensor(snap["res_w_ehs"], device=device, dtype=torch.float32)
            params["res_b_ehs"] = torch.tensor(snap["res_b_ehs"], device=device, dtype=torch.float32)
        if "stream_gate_weights" in snap:
            params["stream_gate_weights"] = torch.tensor(snap["stream_gate_weights"], device=device, dtype=torch.float32)
        if not params and not (zero_hidden_channels or zero_encoder_channels):
            continue
        hook = _BlockAdaLNFixedHook(
            topk_info=topk_per_block[block_idx],
            params=params,
            modulate_hidden=modulate_hidden,
            modulate_encoder=modulate_encoder,
            zero_hidden_channels=zero_hidden_channels,
            zero_encoder_channels=zero_encoder_channels,
            zero_hidden_non_selected=zero_hidden_non_selected,
            zero_encoder_non_selected=zero_encoder_non_selected,
            stream_weight_mode=stream_weight_mode,
            stream_prior_ratio_low=stream_prior_ratio_low,
            stream_prior_ratio_high=stream_prior_ratio_high,
            stream_prior_eps=stream_prior_eps,
        )
        handles.append(block.register_forward_hook(hook))
    return handles

def _compute_iqa_scores(
    sr_tensor: torch.Tensor,
    iqa_metrics: Dict[str, Any],
    ref_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    inp = _sr_to_bchw01(sr_tensor)
    ref_inp = _sr_to_bchw01(ref_tensor) if ref_tensor is not None else None
    out: Dict[str, float] = {}
    with torch.no_grad():
        for name, metric in iqa_metrics.items():
            if name in FR_IQA_METRICS:
                if ref_inp is None:
                    out[name] = float("nan")
                    continue
                score = metric(inp, ref_inp)
            else:
                score = metric(inp)
            if isinstance(score, torch.Tensor):
                out[name] = float(score.detach().float().mean().item())
            else:
                out[name] = float(score)
    return out



def _decode_adaln_predictor_output(
    output: torch.Tensor,
    layout: List[Tuple[int, str, Tuple[int, ...], slice]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    if output.dim() != 2:
        raise ValueError(f"Expected predictor output of shape [B, D], got {tuple(output.shape)}")

    raw_params: Dict[int, Dict[str, torch.Tensor]] = OrderedDict()
    batch_size = int(output.shape[0])
    for block_idx, name, shape, param_slice in layout:
        block_params = raw_params.setdefault(int(block_idx), {})
        if batch_size == 1:
            flat = output[:, param_slice].reshape(shape)
        else:
            flat = output[:, param_slice].reshape((batch_size,) + tuple(shape))
        block_params[name] = flat
    return raw_params


def _sigmoid_range(raw: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    span = max_val - min_val
    return min_val + span * torch.sigmoid(raw)



def _project_param(
    raw: torch.Tensor,
    min_val: float,
    max_val: float,
    constrain_with_sigmoid: bool,
) -> torch.Tensor:
    if constrain_with_sigmoid:
        return _sigmoid_range(raw, min_val, max_val)
    return raw


def _snapshot_from_raw_adaln_tensors(
    raw_params: Dict[int, Dict[str, torch.Tensor]],
    args,
) -> Dict[int, Dict[str, List[float]]]:
    snapshot: Dict[int, Dict[str, List[float]]] = OrderedDict()
    for block_idx in sorted(raw_params.keys()):
        block_out: Dict[str, List[float]] = {}
        block_params = raw_params[block_idx]
        for name, tensor in block_params.items():
            t = tensor.detach().to(torch.float32)
            if name.startswith("gamma_"):
                t = _project_param(
                    t,
                    args.adaln_gamma_min,
                    args.adaln_gamma_max,
                    constrain_with_sigmoid=not args.adaln_free_params,
                )
            elif name.startswith("beta_"):
                t = _project_param(
                    t,
                    args.adaln_beta_min,
                    args.adaln_beta_max,
                    constrain_with_sigmoid=not args.adaln_free_params,
                )
            block_out[name] = t.cpu().tolist()
        if block_out:
            snapshot[int(block_idx)] = block_out
    return snapshot




def _predict_adaln_snapshot_for_image(
    vae: AutoencoderKL,
    predictor: AdaLNVaeConditionPredictor,
    pixel_values: torch.Tensor,
    weight_dtype: torch.dtype,
    args,
    predictor_layout: List[Tuple[int, str, Tuple[int, ...], slice]],
) -> Dict[int, Dict[str, List[float]]]:
    predictor.eval()
    features = _extract_vae_condition_features(
        vae,
        pixel_values,
        weight_dtype=weight_dtype,
        feature_type=str(getattr(args, "adaln_predictor_feature_type", "mean_std")),
    )
    with torch.no_grad():
        predicted = predictor(features.to(device=pixel_values.device, dtype=torch.float32))
    raw_params = _decode_adaln_predictor_output(predicted, predictor_layout)
    return _snapshot_from_raw_adaln_tensors(raw_params, args)

def _build_visual_triptych(
    lr_image: Image.Image,
    sr_before_image: Image.Image,
    sr_after_image: Image.Image,
    panel_size: Tuple[int, int],
) -> Image.Image:
    target_w, target_h = panel_size
    lr_panel = lr_image.resize((target_w, target_h), Image.BICUBIC)
    before_panel = sr_before_image.resize((target_w, target_h), Image.BICUBIC)
    after_panel = sr_after_image.resize((target_w, target_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target_w * 3, target_h), color=(0, 0, 0))
    canvas.paste(lr_panel, (0, 0))
    canvas.paste(before_panel, (target_w, 0))
    canvas.paste(after_panel, (target_w * 2, 0))
    return canvas



def _save_validation_triptychs(
    output_dir: str,
    step_tag: str,
    items: List[Dict[str, Any]],
    max_images: int = 10,
) -> Optional[str]:
    if not items:
        return None

    step_dir = os.path.join(output_dir, "val_set_qualitatives", step_tag)
    os.makedirs(step_dir, exist_ok=True)

    saved = 0
    for item in items:
        lr_image = item.get("lr_image")
        gt_image = item.get("gt_image")
        pred_image = item.get("pred_image")
        if lr_image is None or gt_image is None or pred_image is None:
            continue

        triptych = _build_visual_triptych(
            lr_image=lr_image,
            sr_before_image=gt_image,
            sr_after_image=pred_image,
            panel_size=(int(gt_image.width), int(gt_image.height)),
        )
        stem = os.path.splitext(str(item.get("name", f"sample_{saved:02d}")))[0]
        triptych.save(os.path.join(step_dir, f"{saved:02d}_{stem}.png"))
        saved += 1
        if saved >= max_images:
            break

    return step_dir if saved > 0 else None


def _register_adaln_hooks(
    transformer: SD3Transformer2DModel,
    topk_per_block: Dict[int, Dict[str, Any]],
    raw_params: Dict[int, Dict[str, torch.nn.Parameter]],
    gamma_min: float,
    gamma_max: float,
    beta_min: float,
    beta_max: float,
    modulation_type: str = "residual",
    modulate_hidden: bool = True,
    modulate_encoder: bool = False,
    zero_hidden_channels: bool = False,
    zero_encoder_channels: bool = False,
    zero_hidden_non_selected: bool = False,
    zero_encoder_non_selected: bool = False,
    stream_weight_mode: str = "manual",
    global_stream_gate_logits: Optional[torch.nn.Parameter] = None,
    stream_gate_temperature: float = 1.0,
    stream_prior_ratio_low: float = 0.8,
    stream_prior_ratio_high: float = 1.2,
    stream_prior_eps: float = 1e-6,
    constrain_with_sigmoid: bool = True,
) -> List[Any]:
    handles = []
    if stream_weight_mode == "learned_gate_std_prior_offline":
        _log_std_prior_choices(
            prefix="train-hooks",
            topk_per_block=topk_per_block,
            ratio_low=float(stream_prior_ratio_low),
            ratio_high=float(stream_prior_ratio_high),
            eps=float(stream_prior_eps),
        )
    for block_idx, block in enumerate(transformer.transformer_blocks):
        if block_idx not in topk_per_block:
            continue
        if block_idx not in raw_params:
            continue
        hook = _BlockAdaLNHook(
            block_idx=block_idx,
            topk_info=topk_per_block[block_idx],
            raw_params=raw_params[block_idx],
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            modulation_type=modulation_type,
            modulate_hidden=modulate_hidden,
            modulate_encoder=modulate_encoder,
            zero_hidden_channels=zero_hidden_channels,
            zero_encoder_channels=zero_encoder_channels,
            zero_hidden_non_selected=zero_hidden_non_selected,
            zero_encoder_non_selected=zero_encoder_non_selected,
            stream_weight_mode=stream_weight_mode,
            global_stream_gate_logits=global_stream_gate_logits,
            stream_gate_temperature=stream_gate_temperature,
            stream_prior_ratio_low=stream_prior_ratio_low,
            stream_prior_ratio_high=stream_prior_ratio_high,
            stream_prior_eps=stream_prior_eps,
            constrain_with_sigmoid=constrain_with_sigmoid,
        )
        handles.append(block.register_forward_hook(hook))
    return handles



def _improvement_delta(metric_name: str, before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after):
        return float("nan")
    if metric_name in LOWER_BETTER_METRICS:
        return before - after
    return after - before



def _write_validation_csv(report_dir: str, val_history: List[Dict[str, Any]]) -> Optional[str]:
    rows = [entry for entry in val_history if isinstance(entry, dict) and "scores" in entry]
    if not rows:
        return None

    metric_names: List[str] = []
    for entry in rows:
        scores = entry.get("scores", {})
        for name in scores.keys():
            if name not in metric_names:
                metric_names.append(name)

    csv_path = os.path.join(report_dir, "adaln_validation.csv")
    fieldnames = ["step", "mean_delta", "all_positive"]
    fieldnames += [f"score_{name}" for name in metric_names]
    fieldnames += [f"delta_prev_{name}" for name in metric_names]
    fieldnames += [f"delta_base_{name}" for name in metric_names]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in rows:
            line: Dict[str, Any] = {
                "step": entry.get("step"),
                "mean_delta": entry.get("mean_delta", ""),
                "all_positive": entry.get("all_positive", ""),
            }
            scores = entry.get("scores", {})
            deltas_prev = entry.get("impr_delta_prev", {})
            deltas_base = entry.get("impr_delta_baseline", {})
            for name in metric_names:
                line[f"score_{name}"] = scores.get(name, "")
                line[f"delta_prev_{name}"] = deltas_prev.get(name, "")
                line[f"delta_base_{name}"] = deltas_base.get(name, "")
            writer.writerow(line)

    return csv_path

def _summarize_topk_map(topk_map: Optional[Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    if not isinstance(topk_map, dict) or not topk_map:
        return {
            "num_blocks": 0,
            "hs_total": 0,
            "ehs_total": 0,
            "total": 0,
            "hs_mean_per_block": 0.0,
            "ehs_mean_per_block": 0.0,
            "total_mean_per_block": 0.0,
        }

    hs_total = 0
    ehs_total = 0
    for block_info in topk_map.values():
        if not isinstance(block_info, dict):
            continue
        hs_total += len(block_info.get("hs_topk_idx") or [])
        ehs_total += len(block_info.get("ehs_topk_idx") or [])

    num_blocks = int(len(topk_map))
    total = hs_total + ehs_total
    denom = float(max(1, num_blocks))
    return {
        "num_blocks": num_blocks,
        "hs_total": int(hs_total),
        "ehs_total": int(ehs_total),
        "total": int(total),
        "hs_mean_per_block": float(hs_total / denom),
        "ehs_mean_per_block": float(ehs_total / denom),
        "total_mean_per_block": float(total / denom),
    }
