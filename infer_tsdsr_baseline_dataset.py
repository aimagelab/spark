#!/usr/bin/env python3
"""Run vanilla TSD-SR inference on a dataset and compute IQA metrics.

This is the no-module/no-pipeline baseline path: it loads the released TSD-SR
LoRAs, performs standard inference, and does not register any AdaLN/predictor
hooks.
"""

import argparse
import csv
import json
import math
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
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
]

LOWER_BETTER = {"LPIPS", "DISTS"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run vanilla TSD-SR inference on a named dataset and report LPIPS, "
            "SSIM, MANIQA, CLIPIQA, MUSIQ, and LIQE."
        )
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_ROOTS.keys()))
    parser.add_argument("--dataset_root", default="", help="Optional dataset root override.")
    parser.add_argument("--input_dir", default="", help="Optional LR directory override.")
    parser.add_argument("--gt_dir", default="", help="Optional HR directory override.")
    parser.add_argument("--output_dir", default="", help="Output directory for images and reports.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of evaluated images.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --max_images.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[metric_name for _, metric_name in METRIC_SPECS],
        help="Metrics to compute. Accepts metric names or labels, e.g. lpips SSIM liqe.",
    )

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

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing output images and complete rows from the per-image CSV.",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Compute metrics from existing output images without loading the TSD-SR model.",
    )
    parser.add_argument(
        "--recompute_metrics",
        action="store_true",
        help="Ignore cached per-image CSV rows and recompute metrics.",
    )
    parser.add_argument("--save_every", type=int, default=25, help="Write the per-image CSV every N new rows.")
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


def _normalize_metric_specs(requested_metrics: List[str]) -> List[Tuple[str, str]]:
    by_key: Dict[str, Tuple[str, str]] = {}
    for label, metric_name in METRIC_SPECS:
        by_key[label.lower()] = (label, metric_name)
        by_key[metric_name.lower()] = (label, metric_name)

    selected: List[Tuple[str, str]] = []
    seen = set()
    for raw_name in requested_metrics:
        key = str(raw_name).strip().lower()
        if key not in by_key:
            valid = ", ".join([label for label, _ in METRIC_SPECS])
            raise RuntimeError(f"Unsupported metric '{raw_name}'. Valid metrics: {valid}")
        spec = by_key[key]
        if spec[1] in seen:
            continue
        selected.append(spec)
        seen.add(spec[1])
    if not selected:
        raise RuntimeError("At least one metric must be selected.")
    return selected


def _weight_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _init_eval_metrics(device: str, metric_specs: List[Tuple[str, str]]) -> Dict[str, Any]:
    import pyiqa

    metrics: Dict[str, Any] = OrderedDict()
    for _, metric_name in metric_specs:
        if metric_name == "ssim":
            metrics[metric_name] = pyiqa.create_metric(
                "ssim",
                test_y_channel=True,
                color_space="ycbcr",
                device=device,
            ).eval()
            continue

        candidates = [metric_name]
        if metric_name == "maniqa-pipal":
            candidates = ["maniqa-pipal", "maniqa"]

        loaded = None
        last_exc: Optional[Exception] = None
        for candidate in candidates:
            try:
                loaded = pyiqa.create_metric(candidate, device=device).eval()
                break
            except Exception as exc:
                last_exc = exc
        if loaded is None:
            print(f"[warn] could not initialize metric '{metric_name}': {last_exc}")
            continue
        metrics[metric_name] = loaded
    return metrics


def _read_existing_rows(csv_path: str, metric_specs: List[Tuple[str, str]]) -> Dict[str, Dict[str, Any]]:
    rows_by_name: Dict[str, Dict[str, Any]] = OrderedDict()
    if not csv_path or not os.path.isfile(csv_path):
        return rows_by_name

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            image_name = raw_row.get("image_name", "")
            if not image_name:
                continue
            row: Dict[str, Any] = OrderedDict()
            row["image_name"] = image_name
            row["runtime_sec"] = _safe_float(raw_row.get("runtime_sec"), default=0.0)
            for label, _ in metric_specs:
                row[label] = _safe_float(raw_row.get(label), default=float("nan"))
            rows_by_name[image_name] = row
    return rows_by_name


def _safe_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _row_has_metrics(row: Optional[Dict[str, Any]], metric_specs: List[Tuple[str, str]]) -> bool:
    if not row:
        return False
    for label, _ in metric_specs:
        if label not in row:
            return False
        value = _safe_float(row.get(label), default=float("nan"))
        if not math.isfinite(value):
            return False
    return True


def _write_rows_csv(csv_path: str, rows: List[Dict[str, Any]], metric_specs: List[Tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["image_name", "runtime_sec"] + [label for label, _ in metric_specs]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _pair_rows_to_metric_summary(rows: List[Dict[str, Any]], metric_specs: List[Tuple[str, str]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = OrderedDict()
    summary["num_images"] = len(rows)

    for metric_label, _ in metric_specs:
        values = []
        for row in rows:
            value = _safe_float(row.get(metric_label), default=float("nan"))
            if math.isfinite(value):
                values.append(value)
        direction = "lower" if metric_label in LOWER_BETTER else "higher"
        if not values:
            summary[metric_label] = {"mean": None, "std": None, "direction": direction}
            continue

        tensor = torch.tensor(values, dtype=torch.float64)
        summary[metric_label] = {
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()),
            "direction": direction,
        }

    runtime_values = []
    for row in rows:
        value = _safe_float(row.get("runtime_sec"), default=float("nan"))
        if math.isfinite(value):
            runtime_values.append(value)
    if runtime_values:
        runtime = torch.tensor(runtime_values, dtype=torch.float64)
        summary["runtime_sec"] = {
            "mean": float(runtime.mean().item()),
            "std": float(runtime.std(unbiased=False).item()),
        }
    else:
        summary["runtime_sec"] = {"mean": None, "std": None}
    return summary


def _load_sr_eval_tensor(image_path: str, device: str, weight_dtype: torch.dtype) -> torch.Tensor:
    with Image.open(image_path) as image:
        sr_image = image.convert("RGB").copy()
    return utils._pil_to_tensor_minus1_1(sr_image, device=device, weight_dtype=weight_dtype)


def _write_summary_json(
    summary_json_path: str,
    args: argparse.Namespace,
    dataset_root: str,
    input_dir: str,
    gt_dir: str,
    images_dir: str,
    per_image_csv: str,
    metric_specs: List[Tuple[str, str]],
    rows: List[Dict[str, Any]],
    total_pairs: int,
) -> Dict[str, Any]:
    summary_metrics = _pair_rows_to_metric_summary(rows, metric_specs)
    payload = OrderedDict(
        mode="baseline_no_adaln",
        dataset=args.dataset,
        dataset_root=dataset_root,
        input_dir=input_dir,
        gt_dir=gt_dir,
        output_dir=args.output_dir,
        image_dir=images_dir,
        per_image_csv=per_image_csv,
        total_pairs=total_pairs,
        complete=len(rows) == total_pairs,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        lora_dir=args.lora_dir,
        embedding_dir=args.embedding_dir,
        mixed_precision=args.mixed_precision,
        align_method=args.align_method,
        upscale=args.upscale,
        process_size=args.process_size,
        metrics=summary_metrics,
    )
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _print_summary(prefix: str, summary_metrics: Dict[str, Any], metric_specs: List[Tuple[str, str]]) -> None:
    pretty_metrics = []
    for metric_label, _ in metric_specs:
        stats = summary_metrics.get(metric_label, {})
        mean_val = stats.get("mean")
        if mean_val is None:
            pretty_metrics.append(f"{metric_label}=nan")
        else:
            pretty_metrics.append(f"{metric_label}={mean_val:.6f}")
    print(prefix + " " + " ".join(pretty_metrics))


def main() -> None:
    args = _parse_args()
    utils._set_seed(args.seed)

    metric_specs = _normalize_metric_specs(args.metrics)
    dataset_root, input_dir, gt_dir = _resolve_dataset_dirs(args)

    if not args.output_dir:
        args.output_dir = os.path.join("outputs", f"tsdsr_baseline_eval_{args.dataset}")
    images_dir = os.path.join(args.output_dir, "images")
    reports_dir = os.path.join(args.output_dir, "reports")
    per_image_csv = os.path.join(reports_dir, "baseline_metrics_per_image.csv")
    summary_json_path = os.path.join(reports_dir, "baseline_metrics_summary.json")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

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

    existing_rows = _read_existing_rows(per_image_csv, metric_specs) if args.resume else {}
    reusable_rows = 0
    needs_scoring = False
    needs_inference = False
    for pair in pairs:
        name = pair["name"]
        out_path = os.path.join(images_dir, name)
        cached_row = existing_rows.get(name)
        if args.resume and not args.recompute_metrics and os.path.isfile(out_path) and _row_has_metrics(cached_row, metric_specs):
            reusable_rows += 1
            continue
        needs_scoring = True
        if not args.metrics_only and not os.path.isfile(out_path):
            needs_inference = True

    print(f"[dataset] name={args.dataset}")
    print(f"[dataset] root={dataset_root}")
    print(f"[dataset] lr_dir={input_dir}")
    print(f"[dataset] gt_dir={gt_dir}")
    print(f"[dataset] num_pairs={len(pairs)}")
    print(f"[output] images_dir={images_dir}")
    print(f"[output] per_image_csv={per_image_csv}")
    print(f"[resume] cached_complete_rows={reusable_rows}")
    print("[metrics] " + ", ".join([label for label, _ in metric_specs]))

    weight_dtype = _weight_dtype_from_name(args.mixed_precision)

    transformer = None
    vae = None
    prompt_embeds = None
    pooled_prompt_embeds = None
    timesteps = None
    if needs_inference:
        print("[model] loading vanilla TSD-SR model")
        transformer, vae, prompt_embeds, pooled_prompt_embeds = utils.load_model(args, weight_dtype)
        timesteps = torch.tensor([1000.0], device=args.device, dtype=weight_dtype)
    elif args.metrics_only:
        print("[model] metrics-only mode, model will not be loaded")
    else:
        print("[model] no missing SR images, model will not be loaded")

    iqa_metrics: Dict[str, Any] = {}
    if needs_scoring:
        iqa_metrics = _init_eval_metrics(args.device, metric_specs)
        missing_metrics = [label for label, metric_name in metric_specs if metric_name not in iqa_metrics]
        if missing_metrics:
            print(f"[warn] unavailable metrics will be reported as NaN: {', '.join(missing_metrics)}")
    else:
        print("[metrics] all requested rows are cached")

    rows: List[Dict[str, Any]] = []
    new_rows = 0
    progress = tqdm(pairs, desc=f"TSD-SR baseline {args.dataset}")
    for sample_spec in progress:
        name = sample_spec["name"]
        out_path = os.path.join(images_dir, name)
        cached_row = existing_rows.get(name)

        if args.resume and not args.recompute_metrics and os.path.isfile(out_path) and _row_has_metrics(cached_row, metric_specs):
            rows.append(cached_row)
            continue

        if args.metrics_only and not os.path.isfile(out_path):
            raise RuntimeError(f"Missing output image in metrics-only mode: {out_path}")

        payload = utils._materialize_metric_sample(sample_spec, args, weight_dtype)
        runtime_sec = 0.0

        if os.path.isfile(out_path) and (args.resume or args.metrics_only):
            sr_eval = _load_sr_eval_tensor(out_path, args.device, weight_dtype)
            if cached_row is not None:
                runtime_sec = _safe_float(cached_row.get("runtime_sec"), default=0.0)
        else:
            if transformer is None or vae is None or prompt_embeds is None or pooled_prompt_embeds is None or timesteps is None:
                raise RuntimeError("Model was not loaded, but an SR image needs to be generated.")

            start_time = time.perf_counter()
            sr_tensor = utils._run_sr(
                vae=vae,
                transformer=transformer,
                pixel_values=payload["pixel_values"],
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                weight_dtype=weight_dtype,
                args=args,
                use_no_grad=True,
            )
            sr_image, sr_eval = utils._postprocess_sr_like_classic(
                sr_tensor=sr_tensor,
                lr_image=payload["lr_image"],
                ori_width=payload["ori_width"],
                ori_height=payload["ori_height"],
                upscale=args.upscale,
                resize_flag=payload["resize_flag"],
                align_method=args.align_method,
                device=args.device,
                weight_dtype=weight_dtype,
            )
            runtime_sec = time.perf_counter() - start_time
            sr_image.save(out_path)

        scores_raw = utils._compute_iqa_scores(
            sr_tensor=sr_eval,
            iqa_metrics=iqa_metrics,
            ref_tensor=payload["gt_tensor"],
        )

        row: Dict[str, Any] = OrderedDict()
        row["image_name"] = name
        row["runtime_sec"] = float(runtime_sec)
        for metric_label, metric_name in metric_specs:
            value = scores_raw.get(metric_name, float("nan"))
            row[metric_label] = float(value) if value is not None else float("nan")
        rows.append(row)
        new_rows += 1

        if args.save_every > 0 and new_rows % args.save_every == 0:
            _write_rows_csv(per_image_csv, rows, metric_specs)
            progress.set_postfix(done=len(rows), new=new_rows)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_rows_csv(per_image_csv, rows, metric_specs)
    summary_payload = _write_summary_json(
        summary_json_path=summary_json_path,
        args=args,
        dataset_root=dataset_root,
        input_dir=input_dir,
        gt_dir=gt_dir,
        images_dir=images_dir,
        per_image_csv=per_image_csv,
        metric_specs=metric_specs,
        rows=rows,
        total_pairs=len(pairs),
    )

    _print_summary("[done]", summary_payload["metrics"], metric_specs)
    print(f"[done] images_dir={images_dir}")
    print(f"[done] per_image_csv={per_image_csv}")
    print(f"[done] summary_json={summary_json_path}")


if __name__ == "__main__":
    main()
