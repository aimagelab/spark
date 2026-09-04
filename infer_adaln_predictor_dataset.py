#!/usr/bin/env python3
"""Run AdaLN predictor inference from an experiment folder and evaluate IQA metrics."""

import argparse
import csv
import json
import math
import os
import time
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

import utils.adaln_utils as utils



# ---------------------------------------------------------------------------
# Default asset locations.
#
# Nothing machine-specific is hard-coded: every default is read from an
# environment variable (see README, "Configuration"), and every one of them can
# still be overridden on the command line.
# ---------------------------------------------------------------------------
SD3_MODEL_PATH = os.environ.get(
    "SD3_MODEL_PATH", "checkpoints/stable-diffusion-3-medium-diffusers"
)
TSDSR_LORA_DIR = os.environ.get("TSDSR_LORA_DIR", "checkpoints/tsdsr/lora")
TSDSR_EMBEDDING_DIR = os.environ.get(
    "TSDSR_EMBEDDING_DIR", "checkpoints/tsdsr/embeddings"
)
DATA_ROOT = os.environ.get("DATA_ROOT", "datasets")

DEFAULT_MODEL_ROOT = SD3_MODEL_PATH
DEFAULT_LORA_DIR = TSDSR_LORA_DIR
DEFAULT_EMBEDDING_DIR = TSDSR_EMBEDDING_DIR

DATASET_ROOTS = {
    "DRealSR": os.path.join(DATA_ROOT, "DRealSR"),
    "RealSR": os.path.join(DATA_ROOT, "RealSR"),
    "DIV2K": os.path.join(DATA_ROOT, "DIV2K"),
}

DATASET_SPLITS = {
    "DRealSR": ("test_LR", "test_HR"),
    "RealSR": ("test_LR", "test_HR"),
    "DIV2K": ("lr", "gt"),
}

METRIC_SPECS: List[Tuple[str, str]] = [
    ("LPIPS", "lpips"),
    ("SSIM", "ssim"),
    ("MANIQA", "maniqa-pipal"),
    ("CLIPIQA", "clipiqa"),
    ("MUSIQ", "musiq"),
    ("TOPIQ", "topiq_nr"),
    ("LIQE", "liqe"),
    ("DISTS", "dists"),
    ("DINO", "dinov2_cosine"),
]

LOWER_BETTER = {"LPIPS", "DISTS"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an experiment folder containing reports/adaln_predictor.pt and "
            "reports/adaln_predictor.json, run predictor-based SR on a dataset, and compute IQA metrics."
        )
    )
    parser.add_argument("--experiment_dir", required=True, help="Experiment run directory containing reports/.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_ROOTS.keys()),
        help="Named dataset shortcut.",
    )
    parser.add_argument("--dataset_root", default="", help="Optional dataset root override.")
    parser.add_argument("--input_dir", default="", help="Optional LR directory override.")
    parser.add_argument("--gt_dir", default="", help="Optional HR directory override.")
    parser.add_argument("--output_dir", default="", help="Optional output dir override.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of evaluated images.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --max_images.")

    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--lora_dir", default=DEFAULT_LORA_DIR)
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--rank_vae", type=int, default=64)

    parser.add_argument("--is_use_tile", action="store_true")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--mixed_precision", choices=["fp16", "fp32", "bf16"], default="bf16")
    parser.add_argument("--align_method", choices=["wavelet", "adain", "nofix"], default="wavelet")
    return parser.parse_args()


def _resolve_dataset_dirs(args: argparse.Namespace) -> Tuple[str, str, str]:
    dataset_root = args.dataset_root or DATASET_ROOTS[args.dataset]
    lr_subdir, gt_subdir = DATASET_SPLITS[args.dataset]
    input_dir = args.input_dir or os.path.join(dataset_root, lr_subdir)
    gt_dir = args.gt_dir or os.path.join(dataset_root, gt_subdir)
    if not os.path.isdir(input_dir):
        raise RuntimeError(f"LR directory not found: {input_dir}")
    if not os.path.isdir(gt_dir):
        raise RuntimeError(f"GT directory not found: {gt_dir}")
    return dataset_root, input_dir, gt_dir


def _weight_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_runtime_args(cli_args: argparse.Namespace, predictor_config: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_model_name_or_path=cli_args.pretrained_model_name_or_path,
        lora_dir=cli_args.lora_dir,
        embedding_dir=cli_args.embedding_dir,
        rank=int(cli_args.rank),
        rank_vae=int(cli_args.rank_vae),
        is_use_tile=bool(cli_args.is_use_tile),
        vae_decoder_tiled_size=int(cli_args.vae_decoder_tiled_size),
        vae_encoder_tiled_size=int(cli_args.vae_encoder_tiled_size),
        latent_tiled_size=int(cli_args.latent_tiled_size),
        latent_tiled_overlap=int(cli_args.latent_tiled_overlap),
        device=str(cli_args.device),
        seed=int(cli_args.seed),
        upscale=int(cli_args.upscale),
        process_size=int(cli_args.process_size),
        mixed_precision=str(cli_args.mixed_precision),
        align_method=str(cli_args.align_method),
        adaln_predictor_feature_type=str(predictor_config.get("feature_type", "mean_std")),
        modulate_hidden=bool(predictor_config.get("modulate_hidden", True)),
        modulate_encoder=bool(predictor_config.get("modulate_encoder", False)),
        stream_weight_mode=str(predictor_config.get("stream_weight_mode", "manual")),
        stream_prior_ratio_low=float(predictor_config.get("stream_prior_ratio_low", 0.8)),
        stream_prior_ratio_high=float(predictor_config.get("stream_prior_ratio_high", 1.2)),
        stream_prior_eps=float(predictor_config.get("stream_prior_eps", 1e-6)),
        adaln_gamma_min=float(predictor_config.get("adaln_gamma_min", 0.5)),
        adaln_gamma_max=float(predictor_config.get("adaln_gamma_max", 1.5)),
        adaln_beta_min=float(predictor_config.get("adaln_beta_min", -0.2)),
        adaln_beta_max=float(predictor_config.get("adaln_beta_max", 0.2)),
        adaln_free_params=bool(predictor_config.get("adaln_free_params", False)),
    )


def _load_predictor_assets(experiment_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    report_dir = os.path.join(experiment_dir, "reports")
    summary_path = os.path.join(report_dir, "adaln_predictor.json")
    ckpt_path = os.path.join(report_dir, "adaln_predictor.pt")

    if not os.path.isfile(summary_path):
        raise RuntimeError(f"Predictor summary not found: {summary_path}")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Predictor checkpoint not found: {ckpt_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    return summary, checkpoint, report_dir, summary_path


def _normalize_layout(layout: List[Any]) -> List[Tuple[int, str, Tuple[int, ...], slice]]:
    normalized: List[Tuple[int, str, Tuple[int, ...], slice]] = []
    for item in layout:
        if len(item) != 4:
            raise RuntimeError(f"Unexpected predictor layout item: {item}")
        block_idx, name, shape, sl = item
        if isinstance(sl, slice):
            layout_slice = sl
        elif isinstance(sl, (list, tuple)) and len(sl) >= 2:
            layout_slice = slice(int(sl[0]), int(sl[1]))
        else:
            raise RuntimeError(f"Unexpected layout slice encoding: {sl}")
        normalized.append((int(block_idx), str(name), tuple(shape), layout_slice))
    return normalized


def _init_predictor(
    runtime_args: SimpleNamespace,
    checkpoint: Dict[str, Any],
    vae: Any,
    sample_pixel_values: torch.Tensor,
    weight_dtype: torch.dtype,
) -> Tuple[utils.AdaLNVaeConditionPredictor, List[Tuple[int, str, Tuple[int, ...], slice]]]:
    layout = _normalize_layout(checkpoint["layout"])
    feature_dim = int(
        utils._extract_vae_condition_features(
            vae,
            sample_pixel_values,
            weight_dtype=weight_dtype,
            feature_type=runtime_args.adaln_predictor_feature_type,
        ).shape[-1]
    )
    output_dim = 0
    for _, _, _, sl in layout:
        output_dim = max(output_dim, int(sl.stop))

    predictor_config = checkpoint.get("config", {})
    predictor = utils.AdaLNVaeConditionPredictor(
        input_dim=feature_dim,
        output_dim=output_dim,
        hidden_dim=int(predictor_config.get("hidden_dim", 256)),
        num_layers=int(predictor_config.get("num_layers", 2)),
        dropout=float(predictor_config.get("dropout", 0.0)),
    ).to(device=runtime_args.device, dtype=torch.float32)
    predictor.load_state_dict(checkpoint["state_dict"])
    predictor.eval()
    return predictor, layout


def _pair_rows_to_metric_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = OrderedDict()
    summary["num_images"] = len(rows)

    for metric_label, _ in METRIC_SPECS:
        values = []
        for row in rows:
            value = row.get(metric_label)
            if value is None:
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            values.append(float(value))
        if not values:
            summary[metric_label] = {"mean": None, "std": None, "direction": "lower" if metric_label in LOWER_BETTER else "higher"}
            continue

        tensor = torch.tensor(values, dtype=torch.float64)
        summary[metric_label] = {
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()),
            "direction": "lower" if metric_label in LOWER_BETTER else "higher",
        }
    runtime_values = [float(row["runtime_sec"]) for row in rows if "runtime_sec" in row]
    if runtime_values:
        rt = torch.tensor(runtime_values, dtype=torch.float64)
        summary["runtime_sec"] = {
            "mean": float(rt.mean().item()),
            "std": float(rt.std(unbiased=False).item()),
        }
    else:
        summary["runtime_sec"] = {"mean": None, "std": None}
    return summary


def _write_rows_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["image_name", "runtime_sec"] + [label for label, _ in METRIC_SPECS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _init_eval_metrics(device: str) -> Dict[str, Any]:
    import pyiqa

    metrics: Dict[str, Any] = OrderedDict()
    metrics["lpips"] = pyiqa.create_metric("lpips", device=device).eval()
    metrics["ssim"] = pyiqa.create_metric(
        "ssim",
        test_y_channel=True,
        color_space="ycbcr",
        device=device,
    ).eval()
    metrics["maniqa-pipal"] = pyiqa.create_metric("maniqa-pipal", device=device).eval()
    metrics["clipiqa"] = pyiqa.create_metric("clipiqa", device=device).eval()
    metrics["musiq"] = pyiqa.create_metric("musiq", device=device).eval()
    metrics["topiq_nr"] = pyiqa.create_metric("topiq_nr", device=device).eval()
    metrics["liqe"] = pyiqa.create_metric("liqe", device=device).eval()
    metrics["dists"] = pyiqa.create_metric("dists", device=device).eval()

    dino_metric = utils._init_dinov2_cosine_metric(device)
    if dino_metric is not None:
        metrics["dinov2_cosine"] = dino_metric
    return metrics


def main() -> None:
    args = _parse_args()
    utils._set_seed(args.seed)

    dataset_root, input_dir, gt_dir = _resolve_dataset_dirs(args)
    summary_json, checkpoint, report_dir, summary_path = _load_predictor_assets(args.experiment_dir)
    predictor_config = checkpoint.get("config", {})
    runtime_args = _build_runtime_args(args, predictor_config)
    weight_dtype = _weight_dtype_from_name(runtime_args.mixed_precision)

    output_dir = args.output_dir or os.path.join(args.experiment_dir, f"predictor_eval_{args.dataset}")
    images_dir = os.path.join(output_dir, "images")
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    topk_info = utils._normalize_topk_map(summary_json.get("phase1_topk"))
    if topk_info is None:
        topk_info = utils._normalize_topk_map(checkpoint.get("topk_info"))
    if topk_info is None:
        raise RuntimeError(
            "No phase1 top-k/channel information found in adaln_predictor.json or adaln_predictor.pt."
        )

    pairs_raw = utils._collect_pairs(input_dir, gt_dir)
    if not pairs_raw:
        raise RuntimeError(f"No LR/GT image pairs found in {input_dir}")
    pairs: List[Dict[str, str]] = [
        {"lr_path": item["lr"], "gt_path": item["gt"], "name": os.path.basename(item["lr"])}
        for item in pairs_raw
        if item.get("gt") is not None
    ]
    if not pairs:
        raise RuntimeError(f"No matched GT files found in {gt_dir}")

    if args.shuffle:
        order = torch.randperm(len(pairs)).tolist()
        pairs = [pairs[idx] for idx in order]
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    print(f"[dataset] name={args.dataset}")
    print(f"[dataset] root={dataset_root}")
    print(f"[dataset] lr_dir={input_dir}")
    print(f"[dataset] gt_dir={gt_dir}")
    print(f"[dataset] num_pairs={len(pairs)}")
    print(f"[predictor] summary={summary_path}")

    transformer, vae, prompt_embeds, pooled_prompt_embeds = utils.load_model(runtime_args, weight_dtype)
    timesteps = torch.tensor([1000.0], device=runtime_args.device, dtype=weight_dtype)

    first_payload = utils._materialize_metric_sample(pairs[0], runtime_args, weight_dtype)
    predictor, predictor_layout = _init_predictor(
        runtime_args,
        checkpoint,
        vae,
        first_payload["pixel_values"],
        weight_dtype,
    )

    iqa_metrics = _init_eval_metrics(runtime_args.device)
    missing_metrics = [label for label, metric_name in METRIC_SPECS if metric_name not in iqa_metrics]
    if missing_metrics:
        print(f"[warn] unavailable metrics will be reported as NaN: {', '.join(missing_metrics)}")

    rows: List[Dict[str, Any]] = []
    device = torch.device(runtime_args.device)

    for sample_spec in tqdm(pairs, desc="Predictor inference"):
        payload = utils._materialize_metric_sample(sample_spec, runtime_args, weight_dtype)

        start_time = time.perf_counter()
        adaln_snapshot = utils._predict_adaln_snapshot_for_image(
            vae=vae,
            predictor=predictor,
            pixel_values=payload["pixel_values"],
            weight_dtype=weight_dtype,
            args=runtime_args,
            predictor_layout=predictor_layout,
        )

        handles = utils._register_fixed_adaln_hooks(
            transformer=transformer,
            topk_per_block=topk_info,
            adaln_snapshot=adaln_snapshot,
            device=device,
            modulate_hidden=bool(runtime_args.modulate_hidden),
            modulate_encoder=bool(runtime_args.modulate_encoder),
            stream_weight_mode=str(runtime_args.stream_weight_mode),
            stream_prior_ratio_low=float(runtime_args.stream_prior_ratio_low),
            stream_prior_ratio_high=float(runtime_args.stream_prior_ratio_high),
            stream_prior_eps=float(runtime_args.stream_prior_eps),
        )
        try:
            sr_tensor = utils._run_sr(
                vae=vae,
                transformer=transformer,
                pixel_values=payload["pixel_values"],
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                weight_dtype=weight_dtype,
                args=runtime_args,
                use_no_grad=True,
            )
        finally:
            for handle in handles:
                handle.remove()

        sr_image, sr_eval = utils._postprocess_sr_like_classic(
            sr_tensor=sr_tensor,
            lr_image=payload["lr_image"],
            ori_width=payload["ori_width"],
            ori_height=payload["ori_height"],
            upscale=runtime_args.upscale,
            resize_flag=payload["resize_flag"],
            align_method=runtime_args.align_method,
            device=runtime_args.device,
            weight_dtype=weight_dtype,
        )
        runtime_sec = time.perf_counter() - start_time

        out_path = os.path.join(images_dir, payload["name"])
        sr_image.save(out_path)

        scores_raw = utils._compute_iqa_scores(
            sr_tensor=sr_eval,
            iqa_metrics=iqa_metrics,
            ref_tensor=payload["gt_tensor"],
        )

        row: Dict[str, Any] = OrderedDict()
        row["image_name"] = payload["name"]
        row["runtime_sec"] = float(runtime_sec)
        for metric_label, metric_name in METRIC_SPECS:
            value = scores_raw.get(metric_name, float("nan"))
            row[metric_label] = float(value) if value is not None else float("nan")
        rows.append(row)

    per_image_csv = os.path.join(reports_dir, "predictor_metrics_per_image.csv")
    _write_rows_csv(per_image_csv, rows)

    summary_metrics = _pair_rows_to_metric_summary(rows)
    summary_payload = OrderedDict(
        experiment_dir=args.experiment_dir,
        predictor_report_json=summary_path,
        predictor_checkpoint=os.path.join(report_dir, "adaln_predictor.pt"),
        dataset=args.dataset,
        dataset_root=dataset_root,
        input_dir=input_dir,
        gt_dir=gt_dir,
        output_dir=output_dir,
        image_dir=images_dir,
        per_image_csv=per_image_csv,
        topk_summary=utils._summarize_topk_map(topk_info),
        metrics=summary_metrics,
    )

    summary_json_path = os.path.join(reports_dir, "predictor_metrics_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    pretty_metrics = []
    for metric_label, _ in METRIC_SPECS:
        stats = summary_metrics.get(metric_label, {})
        mean_val = stats.get("mean")
        if mean_val is None:
            pretty_metrics.append(f"{metric_label}=nan")
        else:
            pretty_metrics.append(f"{metric_label}={mean_val:.6f}")
    print("[done] " + " ".join(pretty_metrics))
    print(f"[done] images_dir={images_dir}")
    print(f"[done] per_image_csv={per_image_csv}")
    print(f"[done] summary_json={summary_json_path}")


if __name__ == "__main__":
    main()
