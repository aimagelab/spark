#!/usr/bin/env python3
"""Standalone AdaLN predictor training (no test-time optimization/inference)."""

import json
import os
from typing import Any, Dict, List, Optional
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
import math
from models.autoencoder_kl import AutoencoderKL
import torch
import random
import utils.adaln_utils as utils
from utils.phase1_online import select_channels_online_ema
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

# Local weights only by default (no network calls once the checkpoints are on
# disk). Set HF_ALLOW_DOWNLOAD=1 to let HuggingFace fetch anything missing.
if os.environ.get("HF_ALLOW_DOWNLOAD", "0") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")



def _resolve_alpha_iqa(args) -> float:
    """Weight of the IQA objective.

    --adaln_predictor_alpha_iqa wins when given; otherwise the historical
    --adaln_predictor_alpha_liqe is used, so existing invocations are unchanged.
    """
    alpha = getattr(args, "adaln_predictor_alpha_iqa", None)
    if alpha is None:
        alpha = getattr(args, "adaln_predictor_alpha_liqe", 1.0)
    return float(alpha)

def _validate_training_args(args) -> None:
    if int(args.adaln_predictor_steps) < 1:
        raise RuntimeError("--adaln_predictor_steps must be >= 1.")
    if int(args.adaln_predictor_batch_size) < 1:
        raise RuntimeError("--adaln_predictor_batch_size must be >= 1.")
    if int(getattr(args, "adaln_predictor_grad_accum_steps", 1)) < 1:
        raise RuntimeError("--adaln_predictor_grad_accum_steps must be >= 1.")
    max_num_steps = getattr(args, "adaln_predictor_max_num_steps", None)
    if max_num_steps is not None and int(max_num_steps) < 0:
        raise RuntimeError("--adaln_predictor_max_num_steps must be >= 0 when provided.")
    if float(args.adaln_predictor_lr) <= 0:
        raise RuntimeError("--adaln_predictor_lr must be > 0.")
    if float(getattr(args, "adaln_predictor_alpha_lpips", 0.0)) < 0:
        raise RuntimeError("--adaln_predictor_alpha_lpips must be >= 0.")
    if float(getattr(args, "adaln_predictor_alpha_liqe", 1.0)) < 0:
        raise RuntimeError("--adaln_predictor_alpha_liqe must be >= 0.")
    if _resolve_alpha_iqa(args) < 0:
        raise RuntimeError("--adaln_predictor_alpha_iqa must be >= 0.")
    if float(getattr(args, "adaln_predictor_alpha_tv", 0.0)) < 0:
        raise RuntimeError("--adaln_predictor_alpha_tv must be >= 0.")
    if (
        float(getattr(args, "adaln_predictor_alpha_lpips", 0.0))
        + _resolve_alpha_iqa(args)
        + float(getattr(args, "adaln_predictor_alpha_tv", 0.0))
    ) <= 0:
        raise RuntimeError(
            "At least one of --adaln_predictor_alpha_lpips, --adaln_predictor_alpha_liqe "
            "or --adaln_predictor_alpha_tv must be > 0."
        )
    if int(getattr(args, "adaln_predictor_val_every", 0)) < 0:
        raise RuntimeError("--adaln_predictor_val_every must be >= 0.")
    if int(getattr(args, "adaln_predictor_early_stop_patience", 5)) < 0:
        raise RuntimeError("--adaln_predictor_early_stop_patience must be >= 0.")
    max_num_epochs = getattr(args, "adaln_predictor_max_num_epochs", None)
    if max_num_epochs is not None and int(max_num_epochs) < 1:
        raise RuntimeError("--adaln_predictor_max_num_epochs must be >= 1 when provided.")

    if args.stream_weight_mode != "manual":
        args.modulate_hidden = True
        args.modulate_encoder = True
        print(
            f"[stream] {args.stream_weight_mode} enabled: hidden/encoder stream weights are learned."
        )

    if float(args.stream_gate_temperature) <= 0:
        raise RuntimeError("--stream_gate_temperature must be > 0.")
    if float(args.stream_gate_init_scale) < 0:
        raise RuntimeError("--stream_gate_init_scale must be >= 0.")
    if float(args.stream_prior_ratio_low) <= 0 or float(args.stream_prior_ratio_high) <= 0:
        raise RuntimeError("--stream_prior_ratio_low and --stream_prior_ratio_high must be > 0.")
    if float(args.stream_prior_ratio_low) >= float(args.stream_prior_ratio_high):
        raise RuntimeError("--stream_prior_ratio_low must be < --stream_prior_ratio_high.")
    if float(args.stream_prior_eps) <= 0:
        raise RuntimeError("--stream_prior_eps must be > 0.")


def _force_predictor_only_mode(args) -> None:
    if args.adaln_optimize:
        print("[mode] ignoring --adaln_optimize (this script trains predictor only).")
    if args.tta_enable:
        print("[mode] ignoring --tta_enable (this script trains predictor only).")
    if str(getattr(args, "adaln_predictor_ckpt", "")).strip():
        print("[mode] ignoring --adaln_predictor_ckpt (this script trains a new predictor).")

    args.adaln_optimize = False
    args.tta_enable = False
    args.adaln_predictor_ckpt = ""
    args.adaln_predictor_train = True

def train_adaln_vae_predictor(
    vae: AutoencoderKL,
    transformer: SD3Transformer2DModel,
    opt_samples: List[Dict[str, Any]],
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    weight_dtype: torch.dtype,
    args,
    topk_per_block: Dict[int, Dict[str, Any]],
    liqe_metric: Optional[Any] = None,
    lpips_metric: Optional[Any] = None,
    val_iqa_metrics: Optional[Dict[str, Any]] = None,
    val_samples_override: Optional[List[Dict[str, Any]]] = None,
    val_report_dir: Optional[str] = None,
) -> Tuple[utils.AdaLNVaeConditionPredictor, List[Tuple[int, str, Tuple[int, ...], slice]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opt_samples:
        raise RuntimeError("No optimization samples provided for AdaLN predictor training.")
    alpha_lpips = float(getattr(args, "adaln_predictor_alpha_lpips", 0.0))
    alpha_liqe = _resolve_alpha_iqa(args)
    iqa_metric_name = str(getattr(args, "adaln_predictor_iqa_metric", "liqe"))
    alpha_tv = float(getattr(args, "adaln_predictor_alpha_tv", 0.0))
    tv_norm = str(getattr(args, "adaln_predictor_tv_norm", "l1")).lower()
    if alpha_liqe > 0 and liqe_metric is None:
        raise RuntimeError(
            f"AdaLN predictor training requires the IQA metric ('{iqa_metric_name}') "
            "when its weight is > 0."
        )
    if alpha_lpips > 0 and lpips_metric is None:
        raise RuntimeError("AdaLN predictor training requires lpips_metric when --adaln_predictor_alpha_lpips > 0.")
    if tv_norm not in {"l1", "l2"}:
        raise RuntimeError("--adaln_predictor_tv_norm must be one of: l1, l2.")

    device = torch.device(args.device)
    dummy_sample = utils._materialize_opt_sample(opt_samples[0], args, weight_dtype)
    feature_dim = int(
        utils._extract_vae_condition_features(
            vae,
            dummy_sample["pixel_values"],
            weight_dtype=weight_dtype,
            feature_type=str(getattr(args, "adaln_predictor_feature_type", "mean_std")),
        ).shape[-1]
    )
    del dummy_sample

    param_spec = utils._adaln_param_shape_spec(
        topk_per_block=topk_per_block,
        per_channel=args.adaln_per_channel,
        device=device,
        modulation_type=args.adaln_modulation_type,
        modulate_hidden=args.modulate_hidden,
        modulate_encoder=args.modulate_encoder,
        stream_weight_mode=args.stream_weight_mode,
        constrain_with_sigmoid=not args.adaln_free_params,
    )
    predictor_layout, predictor_output_dim = utils._flatten_adaln_param_spec(param_spec)
    if predictor_output_dim <= 0:
        raise RuntimeError("AdaLN predictor has no output parameters to learn.")
    grad_accum_steps = max(1, int(getattr(args, "adaln_predictor_grad_accum_steps", 1)))

    print(
        "[adaln-structure] "
        f"modulation={args.adaln_modulation_type} per_channel={bool(args.adaln_per_channel)} "
        f"modulate_hidden={bool(args.modulate_hidden)} modulate_encoder={bool(args.modulate_encoder)} "
        f"stream_weight_mode={args.stream_weight_mode} parallel={bool(getattr(args, 'parallel', False))} "
        f"grad_accum_steps={grad_accum_steps}"
    )
    print(
        "[adaln-structure] "
        f"blocks={len(param_spec)} predictor_input_dim={feature_dim} predictor_output_dim={predictor_output_dim}"
    )
    for block_idx in sorted(param_spec.keys()):
        block_spec = param_spec[block_idx]
        topk_info = topk_per_block.get(block_idx, {})
        hs_k = len(list(topk_info.get("hs_topk_idx") or []))
        ehs_k = len(list(topk_info.get("ehs_topk_idx") or []))
        block_param_numel = sum(utils._adaln_param_numel(tuple(shape)) for shape in block_spec.values())
        params_txt = ", ".join([f"{name}:{tuple(shape)}" for name, shape in sorted(block_spec.items())])
        print(
            f"[adaln-structure] block={block_idx} hs_k={hs_k} ehs_k={ehs_k} "
            f"numel={block_param_numel} params=[{params_txt}]"
        )

    predictor = utils.AdaLNVaeConditionPredictor(
        input_dim=feature_dim,
        output_dim=predictor_output_dim,
        hidden_dim=int(getattr(args, "adaln_predictor_hidden_dim", 256)),
        num_layers=int(getattr(args, "adaln_predictor_num_layers", 2)),
        dropout=float(getattr(args, "adaln_predictor_dropout", 0.0)),
    ).to(device=device, dtype=torch.float32)
    last_layer = predictor.net[-1]
    if isinstance(last_layer, torch.nn.Linear):
        print(
            "[adaln-structure] "
            f"predictor_last_layer_in={int(last_layer.in_features)} "
            f"predictor_last_layer_out={int(last_layer.out_features)} "
            f"weight_shape={tuple(last_layer.weight.shape)}"
        )
    optimizer = torch.optim.Adam(predictor.parameters(), lr=float(args.adaln_predictor_lr))

    for p in transformer.parameters():
        p.requires_grad = False
    for p in vae.parameters():
        p.requires_grad = False
    transformer.eval()
    vae.eval()

    val_history: List[Dict[str, Any]] = []
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_step: Optional[int] = None
    best_mean_delta = -float("inf")
    val_every = max(0, int(getattr(args, "adaln_predictor_val_every", 0)))
    val_max_samples = max(1, int(getattr(args, "adaln_predictor_val_max_samples", 4)))
    val_source_samples = val_samples_override if val_samples_override else opt_samples
    val_samples = val_source_samples[:val_max_samples]
    train_batch_size = max(1, min(int(getattr(args, "adaln_predictor_batch_size", 1)), len(opt_samples)))
    parallel_mode = bool(getattr(args, "parallel", False))
    early_stop_patience = max(0, int(getattr(args, "adaln_predictor_early_stop_patience", 5)))
    no_improve_validations = 0
    best_val_loss = float("inf")
    best_val_loss_step: Optional[int] = None
    last_validation_completed_steps = 0
    stop_reason = "completed"
    current_validation_step_tag = "init"
    shuffled_indices = list(range(len(opt_samples)))
    random.shuffle(shuffled_indices)
    sample_ptr = 0

    def _next_batch() -> List[Dict[str, Any]]:
        nonlocal shuffled_indices, sample_ptr
        batch: List[Dict[str, Any]] = []
        while len(batch) < train_batch_size:
            if sample_ptr >= len(shuffled_indices):
                random.shuffle(shuffled_indices)
                sample_ptr = 0
            batch.append(opt_samples[shuffled_indices[sample_ptr]])
            sample_ptr += 1
        return batch

    def _expand_conditioning(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if int(tensor.shape[0]) == batch_size:
            return tensor
        if int(tensor.shape[0]) == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        raise RuntimeError(
            f"Cannot expand conditioning tensor from batch {int(tensor.shape[0])} to {batch_size}."
        )

    def _weighted_predictor_loss(
        sr_01: torch.Tensor,
        gt_tensor: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        def _tv_loss(x: torch.Tensor, norm: str) -> torch.Tensor:
            if x.ndim != 4:
                raise RuntimeError(f"TV loss expects BCHW tensor, got shape={tuple(x.shape)}.")
            dh = x[:, :, 1:, :] - x[:, :, :-1, :]
            dw = x[:, :, :, 1:] - x[:, :, :, :-1]
            if norm == "l2":
                return dh.square().mean() + dw.square().mean()
            return dh.abs().mean() + dw.abs().mean()

        loss = torch.zeros((), device=sr_01.device, dtype=torch.float32)
        details: Dict[str, float] = {
            "loss_total": 0.0,
            "loss_lpips": 0.0,
            "loss_liqe": 0.0,
            "loss_tv": 0.0,
        }
        if alpha_lpips > 0:
            if gt_tensor is None:
                raise RuntimeError(
                    "LPIPS predictor loss requested (alpha_lpips > 0) but GT tensor is missing. "
                    "Provide --opt_gt_dir and matching GT images."
                )
            gt_01 = utils._sr_to_bchw01(gt_tensor).to(device=sr_01.device, dtype=torch.float32)
            loss_lpips = lpips_metric(sr_01.float(), gt_01)
            loss = loss + (alpha_lpips * loss_lpips)
            details["loss_lpips"] = float(loss_lpips.detach().item())
        if alpha_liqe > 0:
            # Every supported IQA metric is higher-is-better, so the loss is -metric.
            loss_iqa = -liqe_metric(sr_01.float())
            loss = loss + (alpha_liqe * loss_iqa)
            details["loss_iqa"] = float(loss_iqa.detach().item())
            if iqa_metric_name == "liqe":
                details["loss_liqe"] = details["loss_iqa"]
        if alpha_tv > 0:
            loss_tv = _tv_loss(sr_01.float(), tv_norm)
            loss = loss + (alpha_tv * loss_tv)
            details["loss_tv"] = float(loss_tv.detach().item())
        details["loss_total"] = float(loss.detach().item())
        return loss, details

    def _evaluate_predictor_validation() -> Optional[Dict[str, Any]]:
        predictor.eval()
        summary = {name: {"sum": 0.0, "count": 0} for name in (val_iqa_metrics.keys() if val_iqa_metrics else [])}
        val_loss_sum = 0.0
        val_loss_count = 0
        qualitative_items: List[Dict[str, Any]] = []
        for sample in val_samples:
            payload = utils._materialize_metric_sample(sample, args, weight_dtype)
            snapshot = utils._predict_adaln_snapshot_for_image(
                vae,
                predictor,
                payload["pixel_values"],
                weight_dtype,
                args,
                predictor_layout,
            )
            handles = utils._register_fixed_adaln_hooks(
                transformer,
                topk_per_block,
                snapshot,
                device=device,
                modulate_hidden=args.modulate_hidden,
                modulate_encoder=args.modulate_encoder,
                zero_hidden_channels=False,
                zero_encoder_channels=False,
                stream_weight_mode=args.stream_weight_mode,
                stream_prior_ratio_low=float(args.stream_prior_ratio_low),
                stream_prior_ratio_high=float(args.stream_prior_ratio_high),
                stream_prior_eps=float(args.stream_prior_eps),
            )
            try:
                sr = utils._run_sr(
                    vae,
                    transformer,
                    payload["pixel_values"],
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
            sr_01 = utils._sr_to_bchw01(sr)
            val_loss, _ = _weighted_predictor_loss(sr_01, payload.get("gt_tensor"))
            val_loss_sum += float(val_loss.detach().item())
            val_loss_count += 1
            _, sr_eval = utils._postprocess_sr_like_classic(
                sr,
                payload["lr_image"],
                payload["ori_width"],
                payload["ori_height"],
                args.upscale,
                payload["resize_flag"],
                args.align_method,
                args.device,
                weight_dtype,
            )
            if payload.get("gt_image") is not None and len(qualitative_items) < 10:
                qualitative_items.append(
                    {
                        "name": payload["name"],
                        "lr_image": payload["lr_image"],
                        "gt_image": payload["gt_image"],
                        "pred_image": utils._tensor_to_pil(sr_eval),
                    }
                )
            if val_iqa_metrics:
                scores = utils._compute_iqa_scores(sr_eval, val_iqa_metrics, ref_tensor=payload["gt_tensor"])
                for name, value in scores.items():
                    if math.isfinite(value):
                        summary[name]["sum"] += float(value)
                        summary[name]["count"] += 1
        utils._save_validation_triptychs(
            output_dir=args.output_dir,
            step_tag=f"predictor_step_{current_validation_step_tag}",
            items=qualitative_items,
            max_images=10,
        )
        out_scores: Dict[str, float] = {}
        for name, stats in summary.items():
            if stats["count"] > 0:
                out_scores[name] = stats["sum"] / stats["count"]
        if val_loss_count <= 0:
            return None
        return {
            "scores": out_scores,
            "val_loss": val_loss_sum / float(val_loss_count),
        }

    def _evaluate_plain_validation() -> Optional[Dict[str, float]]:
        if not val_iqa_metrics:
            return None
        summary = {name: {"sum": 0.0, "count": 0} for name in val_iqa_metrics.keys()}
        qualitative_items: List[Dict[str, Any]] = []
        for sample in val_samples:
            payload = utils._materialize_metric_sample(sample, args, weight_dtype)
            sr = utils._run_sr(
                vae,
                transformer,
                payload["pixel_values"],
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                weight_dtype,
                args,
                use_no_grad=True,
            )
            _, sr_eval = utils._postprocess_sr_like_classic(
                sr,
                payload["lr_image"],
                payload["ori_width"],
                payload["ori_height"],
                args.upscale,
                payload["resize_flag"],
                args.align_method,
                args.device,
                weight_dtype,
            )
            if payload.get("gt_image") is not None and len(qualitative_items) < 10:
                qualitative_items.append(
                    {
                        "name": payload["name"],
                        "lr_image": payload["lr_image"],
                        "gt_image": payload["gt_image"],
                        "pred_image": utils._tensor_to_pil(sr_eval),
                    }
                )
            scores = utils._compute_iqa_scores(sr_eval, val_iqa_metrics, ref_tensor=payload["gt_tensor"])
            for name, value in scores.items():
                if math.isfinite(value):
                    summary[name]["sum"] += float(value)
                    summary[name]["count"] += 1
        utils._save_validation_triptychs(
            output_dir=args.output_dir,
            step_tag="predictor_step_baseline",
            items=qualitative_items,
            max_images=10,
        )
        out: Dict[str, float] = {}
        for name, stats in summary.items():
            if stats["count"] > 0:
                out[name] = stats["sum"] / stats["count"]
        return out if out else None

    def _flush_validation_csv() -> None:
        if not val_report_dir:
            return
        try:
            utils._write_validation_csv(val_report_dir, val_history)
        except Exception as exc:
            print(f"[warn] could not write predictor validation CSV: {exc}", flush=True)

    def _record_validation_result(
        val_payload: Dict[str, Any],
        *,
        step: int,
        mode: str = "scheduled",
    ) -> None:
        nonlocal best_state_dict, best_step, best_val_loss, best_val_loss_step
        nonlocal no_improve_validations, best_mean_delta, last_validation_completed_steps

        val_scores = dict(val_payload.get("scores") or {})
        val_loss = float(val_payload["val_loss"])
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_step = step
            no_improve_validations = 0
            best_step = step
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in predictor.state_dict().items()
            }
        else:
            no_improve_validations += 1

        mean_delta = -float("inf")
        if baseline_scores is not None:
            finite_deltas = [
                utils._improvement_delta(name, baseline_scores[name], val_scores[name])
                for name in val_scores.keys()
                if name in baseline_scores and math.isfinite(val_scores[name]) and math.isfinite(baseline_scores[name])
            ]
            if finite_deltas:
                mean_delta = sum(finite_deltas) / len(finite_deltas)
        if mean_delta > best_mean_delta:
            best_mean_delta = mean_delta

        val_entry: Dict[str, Any] = {
            "step": step,
            "scores": val_scores,
            "mean_delta": mean_delta,
            "val_loss": val_loss,
        }
        if mode != "scheduled":
            val_entry["mode"] = mode
        val_history.append(val_entry)
        last_validation_completed_steps = max(last_validation_completed_steps, step + 1)
        _flush_validation_csv()

        msg = (
            f"[adaln-predictor-val] step={step} val_loss={val_loss:.6f} "
            f"best_val_loss={best_val_loss:.6f} no_improve={no_improve_validations}/{early_stop_patience}"
        )
        if mode != "scheduled":
            msg += f" mode={mode}"
        if val_scores:
            msg += " " + " ".join([f"{k}={v:.4f}" for k, v in val_scores.items()])
        print(msg, flush=True)

    baseline_scores = _evaluate_plain_validation() if (val_every > 0 and val_iqa_metrics) else None
    if baseline_scores is not None:
        val_history.append({"step": -1, "scores": baseline_scores, "mode": "baseline_no_adaln"})
        _flush_validation_csv()
        print(
            "[adaln-predictor-val] step=-1 (baseline) "
            + " ".join([f"{k}={v:.4f}" for k, v in baseline_scores.items()]),
            flush=True,
        )

    configured_steps = int(args.adaln_predictor_steps)
    max_num_steps_arg = getattr(args, "adaln_predictor_max_num_steps", None)
    max_num_steps = None if max_num_steps_arg is None or int(max_num_steps_arg) <= 0 else int(max_num_steps_arg)
    train_steps = min(configured_steps, max_num_steps) if max_num_steps is not None else configured_steps
    max_num_epochs_arg = getattr(args, "adaln_predictor_max_num_epochs", None)
    max_num_epochs = None if max_num_epochs_arg is None else int(max_num_epochs_arg)
    samples_per_opt_step = max(1, train_batch_size * grad_accum_steps)
    steps_per_epoch = max(1, int(math.ceil(float(len(opt_samples)) / float(samples_per_opt_step))))
    total_seen_samples = 0
    completed_steps = 0
    completed_epochs = 0.0
    train_pbar = tqdm(range(train_steps), desc="adaln-predictor-train", dynamic_ncols=True)
    for step in train_pbar:
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_loss_count = 0
        last_micro_batch_size = train_batch_size
        for _ in range(grad_accum_steps):
            step_batch = _next_batch()
            last_micro_batch_size = len(step_batch)
            if parallel_mode:
                payloads = [utils._materialize_opt_sample(sample, args, weight_dtype) for sample in step_batch]
                grouped_payloads: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
                for payload in payloads:
                    shape_key = tuple(payload["pixel_values"].shape[-2:])
                    grouped_payloads.setdefault(shape_key, []).append(payload)

                for group in grouped_payloads.values():
                    group_pixel_values = torch.cat([p["pixel_values"] for p in group], dim=0)
                    group_prompt_embeds = _expand_conditioning(prompt_embeds, group_pixel_values.shape[0])
                    group_pooled_prompt_embeds = _expand_conditioning(pooled_prompt_embeds, group_pixel_values.shape[0])

                    features = utils._extract_vae_condition_features(
                        vae,
                        group_pixel_values,
                        weight_dtype=weight_dtype,
                        feature_type=str(getattr(args, "adaln_predictor_feature_type", "mean_std")),
                    )
                    predicted = predictor(features.to(device=device, dtype=torch.float32))
                    raw_params = utils._decode_adaln_predictor_output(predicted, predictor_layout)
                    handles = utils._register_adaln_hooks(
                        transformer,
                        topk_per_block,
                        raw_params,
                        gamma_min=args.adaln_gamma_min,
                        gamma_max=args.adaln_gamma_max,
                        beta_min=args.adaln_beta_min,
                        beta_max=args.adaln_beta_max,
                        modulation_type=args.adaln_modulation_type,
                        modulate_hidden=args.modulate_hidden,
                        modulate_encoder=args.modulate_encoder,
                        zero_hidden_channels=False,
                        zero_encoder_channels=False,
                        stream_weight_mode=args.stream_weight_mode,
                        global_stream_gate_logits=None,
                        stream_gate_temperature=float(getattr(args, "stream_gate_temperature", 1.0)),
                        stream_prior_ratio_low=float(getattr(args, "stream_prior_ratio_low", 0.8)),
                        stream_prior_ratio_high=float(getattr(args, "stream_prior_ratio_high", 1.2)),
                        stream_prior_eps=float(getattr(args, "stream_prior_eps", 1e-6)),
                        constrain_with_sigmoid=not args.adaln_free_params,
                    )
                    try:
                        sr = utils._run_sr(
                            vae,
                            transformer,
                            group_pixel_values,
                            timesteps,
                            group_prompt_embeds,
                            group_pooled_prompt_embeds,
                            weight_dtype,
                            args,
                            use_no_grad=False,
                        )
                    finally:
                        for h in handles:
                            h.remove()

                    sr_01 = utils._sr_to_bchw01(sr)
                    group_gt = None
                    if alpha_lpips > 0:
                        gt_tensors = [p.get("gt_tensor") for p in group]
                        if any(g is None for g in gt_tensors):
                            raise RuntimeError(
                                "LPIPS predictor loss requested (alpha_lpips > 0) but GT is missing in optimization samples."
                            )
                        group_gt = torch.stack([g for g in gt_tensors], dim=0)
                    loss, loss_details = _weighted_predictor_loss(sr_01, group_gt)
                    if not loss.requires_grad:
                        raise RuntimeError("AdaLN predictor weighted loss does not require gradients.")
                    group_weight = float(len(group)) / float(len(step_batch))
                    (loss * group_weight / float(grad_accum_steps)).backward()
                    running_loss += float(loss_details["loss_total"]) * float(len(group))
                    running_loss_count += len(group)
            else:
                for sample in step_batch:
                    payload = utils._materialize_opt_sample(sample, args, weight_dtype)
                    features = utils._extract_vae_condition_features(
                        vae,
                        payload["pixel_values"],
                        weight_dtype=weight_dtype,
                        feature_type=str(getattr(args, "adaln_predictor_feature_type", "mean_std")),
                    )
                    predicted = predictor(features.to(device=device, dtype=torch.float32))
                    raw_params = utils._decode_adaln_predictor_output(predicted, predictor_layout)
                    handles = utils._register_adaln_hooks(
                        transformer,
                        topk_per_block,
                        raw_params,
                        gamma_min=args.adaln_gamma_min,
                        gamma_max=args.adaln_gamma_max,
                        beta_min=args.adaln_beta_min,
                        beta_max=args.adaln_beta_max,
                        modulation_type=args.adaln_modulation_type,
                        modulate_hidden=args.modulate_hidden,
                        modulate_encoder=args.modulate_encoder,
                        zero_hidden_channels=False,
                        zero_encoder_channels=False,
                        stream_weight_mode=args.stream_weight_mode,
                        global_stream_gate_logits=None,
                        stream_gate_temperature=float(getattr(args, "stream_gate_temperature", 1.0)),
                        stream_prior_ratio_low=float(getattr(args, "stream_prior_ratio_low", 0.8)),
                        stream_prior_ratio_high=float(getattr(args, "stream_prior_ratio_high", 1.2)),
                        stream_prior_eps=float(getattr(args, "stream_prior_eps", 1e-6)),
                        constrain_with_sigmoid=not args.adaln_free_params,
                    )
                    try:
                        sr = utils._run_sr(
                            vae,
                            transformer,
                            payload["pixel_values"],
                            timesteps,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            weight_dtype,
                            args,
                            use_no_grad=False,
                        )
                    finally:
                        for h in handles:
                            h.remove()

                    sr_01 = utils._sr_to_bchw01(sr)
                    loss, loss_details = _weighted_predictor_loss(sr_01, payload.get("gt_tensor"))
                    if not loss.requires_grad:
                        raise RuntimeError("AdaLN predictor weighted loss does not require gradients.")
                    (loss / float(len(step_batch)) / float(grad_accum_steps)).backward()
                    running_loss += float(loss_details["loss_total"])
                    running_loss_count += 1

        optimizer.step()
        completed_steps = step + 1
        total_seen_samples += int(running_loss_count)
        completed_epochs = float(total_seen_samples) / float(max(1, len(opt_samples)))
        mean_loss = running_loss / float(max(1, running_loss_count))
        train_pbar.set_postfix(loss=f"{mean_loss:.4f}")
        if args.log_every > 0 and (step % args.log_every == 0):
            print(
                f"[adaln-predictor] step={step} loss={mean_loss:.6f} "
                f"batch_size={last_micro_batch_size} grad_accum_steps={grad_accum_steps} "
                f"effective_batch_size={last_micro_batch_size * grad_accum_steps}"
            , flush=True)

        if val_every > 0 and ((step + 1) % val_every == 0):
            current_validation_step_tag = str(step)
            val_payload = _evaluate_predictor_validation()
            if val_payload is not None:
                _record_validation_result(val_payload, step=step)
                train_pbar.set_postfix(loss=f"{mean_loss:.4f}", val_loss=f"{float(val_payload['val_loss']):.4f}")
                if early_stop_patience > 0 and no_improve_validations >= early_stop_patience:
                    stop_reason = f"early_stop_no_improve_{early_stop_patience}"
                    print(
                        f"[adaln-predictor] early stopping at step={step} after "
                        f"{no_improve_validations} validation checks without val_loss improvement."
                    , flush=True)
                    break
        if max_num_epochs is not None and completed_epochs >= float(max_num_epochs):
            stop_reason = "max_num_epochs"
            print(
                f"[adaln-predictor] stopping at step={step} because max_num_epochs={max_num_epochs} was reached "
                f"(completed_epochs={completed_epochs:.4f}).",
                flush=True,
            )
            break

    final_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in predictor.state_dict().items()
    }

    if stop_reason == "completed" and completed_steps >= train_steps:
        if max_num_steps is not None and train_steps == max_num_steps:
            stop_reason = "max_num_steps"
        else:
            stop_reason = "max_steps"
    if val_every > 0 and completed_steps > last_validation_completed_steps:
        final_eval_step = completed_steps - 1
        current_validation_step_tag = f"{final_eval_step}_final"
        print(
            f"[adaln-predictor] running final validation before exit at step={final_eval_step} "
            f"(last_eval_completed_steps={last_validation_completed_steps}, completed_steps={completed_steps}).",
            flush=True,
        )
        final_val_payload = _evaluate_predictor_validation()
        if final_val_payload is not None:
            _record_validation_result(final_val_payload, step=final_eval_step, mode="final_before_exit")
    if best_state_dict is not None:
        predictor.load_state_dict(best_state_dict)

    selection_info = {
        "selected_step": best_step,
        "selected_mean_delta": best_mean_delta,
        "best_val_loss": best_val_loss,
        "best_val_loss_step": best_val_loss_step,
        "stop_reason": stop_reason,
        "configured_steps": configured_steps,
        "max_steps": train_steps,
        "max_num_steps": max_num_steps,
        "max_num_epochs": max_num_epochs,
        "steps_per_epoch": steps_per_epoch,
        "completed_steps": completed_steps,
        "completed_epochs": completed_epochs,
        "feature_dim": feature_dim,
        "output_dim": predictor_output_dim,
        "best_state_dict": best_state_dict,
        "last_state_dict": final_state_dict,
    }
    return predictor, predictor_layout, val_history, selection_info


def main() -> None:
    args = utils.parse_args()
    _force_predictor_only_mode(args)

    print("\n" + "=" * 80)
    print("CLI Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80 + "\n")

    _validate_training_args(args)
    utils._set_seed(args.seed)

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    import pyiqa

    liqe_metric = None
    if _resolve_alpha_iqa(args) > 0:
        iqa_name = str(getattr(args, "adaln_predictor_iqa_metric", "liqe"))
        iqa_kwargs = {}
        if iqa_name.startswith("maniqa"):
            # In eval mode MANIQA averages test_sample uniform crops; each keeps a
            # ViT graph alive, so the default 20 OOMs a 45G card during training.
            iqa_kwargs["test_sample"] = int(
                getattr(args, "adaln_predictor_maniqa_test_sample", 20)
            )
        liqe_metric = pyiqa.create_metric(
            iqa_name, device=args.device, as_loss=True, **iqa_kwargs
        )
        liqe_metric.eval()
        if iqa_kwargs:
            print(f"[adaln-predictor] IQA metric kwargs: {iqa_kwargs}")
        # MANIQA's eval path scores 20 uniform 224x224 crops per image and keeps
        # every one of them in the autograd graph, which OOMs a 45G card even at
        # batch_size 2.  Its own training path uses a single crop, so as a loss we
        # follow that.  Only the objective is affected: the reported MANIQA metric
        # comes from the eval scripts, which build their own untouched instance.
        if iqa_name.startswith("maniqa"):
            crops = int(getattr(args, "adaln_predictor_maniqa_crops", 1))
            net = getattr(liqe_metric, "net", None)
            if net is not None and hasattr(net, "test_sample"):
                net.test_sample = crops
                print(f"[adaln-predictor] MANIQA loss crops per image: {crops}")
        print(
            f"[adaln-predictor] IQA objective: {iqa_name} "
            f"(alpha={_resolve_alpha_iqa(args)})"
        )

    lpips_metric = None
    if float(getattr(args, "adaln_predictor_alpha_lpips", 0.0)) > 0:
        lpips_metric = pyiqa.create_metric("lpips", device=args.device, as_loss=True)
        lpips_metric.eval()

    val_iqa_metrics = None
    if int(args.adaln_predictor_val_every) > 0:
        iqa_metric_names = [
            "maniqa-pipal",
            "musiq",
            "clipiqa",
            "liqe",
            "niqe",
            "lpips",
            "dists",
            "psnr",
            "ssim",
            "dinov2_cosine",
        ]
        val_iqa_metrics = utils._init_iqa_metrics(args.device, iqa_metric_names)

    os.makedirs(args.output_dir, exist_ok=True)
    report_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    infer_pairs = utils._collect_pairs(args.input_dir, args.gt_dir)
    if not infer_pairs:
        raise RuntimeError(f"No inference input images found in {args.input_dir}")

    opt_pairs = utils._collect_pairs(args.opt_input_dir, args.opt_gt_dir)
    if not opt_pairs:
        raise RuntimeError(f"No optimization images found in {args.opt_input_dir}")

    transformer, vae, prompt_embeds, pooled_prompt_embeds = utils.load_model(args, weight_dtype)
    timesteps = torch.tensor([1000.0], device=args.device, dtype=weight_dtype)
    num_transformer_blocks = len(transformer.transformer_blocks)

    zero_out_mode = args.zero_out_selection_mode
    topk_cache = args.topk_cache_file
    if topk_cache is not None and zero_out_mode != "random":
        if not os.path.exists(topk_cache):
            raise RuntimeError(f"Cached topk file not found: {topk_cache}")
        global_topk_info, _ = utils._load_cached_topk(topk_cache)
        print(f"[phase1-topk] loaded cached topk from {topk_cache}")
    else:
        if topk_cache is not None and zero_out_mode == "random":
            print("[phase1-topk] --topk_cache_file ignored for random selection_mode.")
        phase1_pairs = utils._select_phase1_pairs(
            opt_pairs,
            max_samples=args.phase1_max_samples,
            seed=args.seed,
            shuffle=args.phase1_shuffle_samples,
        )
        if len(phase1_pairs) != len(opt_pairs):
            print(
                f"[phase1-topk] using subset: {len(phase1_pairs)}/{len(opt_pairs)} "
                f"(shuffle={args.phase1_shuffle_samples})"
            )
        if str(getattr(args, "phase1_selector", "online_ema")) == "online_ema":
            # Paper Phase 1 (Algorithm 1): online EMA ranking, stopped early by
            # the Jaccard stability criterion. --phase1_max_samples is only an
            # upper budget here; the scan usually terminates well before it.
            global_topk_info = select_channels_online_ema(
                phase1_pairs,
                vae,
                transformer,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                weight_dtype,
                args,
            )
        else:
            # Full-pass selector: scores every image, then ranks once.
            global_topk_info = utils._aggregate_topk_from_dataset(
                phase1_pairs,
                vae,
                transformer,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                weight_dtype,
                args,
                phase1_nr_metric=None,
                phase1_lpips_metric=None,
                include_score_vectors=False,
            )

    global_topk_info = utils._filter_topk_by_layer_scope(
        global_topk_info,
        num_transformer_blocks,
        args.layer_scope,
    )
    if global_topk_info is None or len(global_topk_info) == 0:
        raise RuntimeError("Top-k extraction produced no blocks after layer_scope filtering.")

    opt_samples = utils._prepare_optimization_samples(opt_pairs, args, weight_dtype)
    val_metric_samples = utils._prepare_metric_samples(
        infer_pairs,
        args,
        weight_dtype,
        max_samples=args.adaln_predictor_val_max_samples,
    )

    predictor, predictor_layout, val_history, selection_info = train_adaln_vae_predictor(
        vae,
        transformer,
        opt_samples,
        timesteps,
        prompt_embeds,
        pooled_prompt_embeds,
        weight_dtype,
        args,
        global_topk_info,
        liqe_metric=liqe_metric,
        lpips_metric=lpips_metric,
        val_iqa_metrics=val_iqa_metrics,
        val_samples_override=val_metric_samples,
        val_report_dir=report_dir,
    )

    checkpoint_config = {
        "hidden_dim": int(args.adaln_predictor_hidden_dim),
        "num_layers": int(args.adaln_predictor_num_layers),
        "dropout": float(args.adaln_predictor_dropout),
        "feature_type": str(args.adaln_predictor_feature_type),
        "per_channel": bool(args.adaln_per_channel),
        "modulation_type": str(args.adaln_modulation_type),
        "modulate_hidden": bool(args.modulate_hidden),
        "modulate_encoder": bool(args.modulate_encoder),
        "stream_weight_mode": str(args.stream_weight_mode),
        "adaln_gamma_min": float(args.adaln_gamma_min),
        "adaln_gamma_max": float(args.adaln_gamma_max),
        "adaln_beta_min": float(args.adaln_beta_min),
        "adaln_beta_max": float(args.adaln_beta_max),
        "adaln_free_params": bool(args.adaln_free_params),
        "predictor_alpha_lpips": float(getattr(args, "adaln_predictor_alpha_lpips", 0.0)),
        "predictor_alpha_liqe": float(getattr(args, "adaln_predictor_alpha_liqe", 1.0)),
        "predictor_iqa_metric": str(getattr(args, "adaln_predictor_iqa_metric", "liqe")),
        "predictor_alpha_iqa": _resolve_alpha_iqa(args),
        "predictor_alpha_tv": float(getattr(args, "adaln_predictor_alpha_tv", 0.0)),
        "predictor_tv_norm": str(getattr(args, "adaln_predictor_tv_norm", "l1")),
        "predictor_early_stop_patience": int(getattr(args, "adaln_predictor_early_stop_patience", 5)),
        "predictor_max_num_steps": getattr(args, "adaln_predictor_max_num_steps", None),
        "predictor_max_num_epochs": getattr(args, "adaln_predictor_max_num_epochs", None),
    }
    best_state_dict = selection_info.pop("best_state_dict", None)
    last_state_dict = selection_info.pop("last_state_dict", None)
    predictor_ckpt_path = os.path.join(report_dir, "adaln_predictor.pt")
    predictor_best_ckpt_path = os.path.join(report_dir, "adaln_predictor_best.pt")
    predictor_last_ckpt_path = os.path.join(report_dir, "adaln_predictor_last.pt")
    checkpoint_payload = {
        "layout": predictor_layout,
        "topk_info": global_topk_info,
        "config": checkpoint_config,
    }
    if best_state_dict is not None:
        torch.save({**checkpoint_payload, "state_dict": best_state_dict}, predictor_best_ckpt_path)
        torch.save({**checkpoint_payload, "state_dict": best_state_dict}, predictor_ckpt_path)
    else:
        torch.save({**checkpoint_payload, "state_dict": predictor.state_dict()}, predictor_ckpt_path)
    if last_state_dict is not None:
        torch.save({**checkpoint_payload, "state_dict": last_state_dict}, predictor_last_ckpt_path)

    val_csv = utils._write_validation_csv(report_dir, val_history)
    summary: Dict[str, Any] = {
        "opt_input_dir": args.opt_input_dir,
        "opt_gt_dir": args.opt_gt_dir,
        "val_input_dir": args.input_dir,
        "val_gt_dir": args.gt_dir,
        "num_opt_images": len(opt_pairs),
        "num_val_images": len(val_metric_samples),
        "phase1_topk": global_topk_info,
        "num_topk_blocks": len(global_topk_info),
        "predictor_checkpoint": predictor_ckpt_path,
        "predictor_best_checkpoint": predictor_best_ckpt_path if best_state_dict is not None else predictor_ckpt_path,
        "predictor_last_checkpoint": predictor_last_ckpt_path if last_state_dict is not None else None,
        "predictor_selection": selection_info,
        "predictor_validation_csv": val_csv,
        "predictor_validation": val_history,
        "predictor_feature_type": args.adaln_predictor_feature_type,
        "predictor_steps": args.adaln_predictor_steps,
        "predictor_max_num_steps": getattr(args, "adaln_predictor_max_num_steps", None),
        "predictor_max_num_epochs": getattr(args, "adaln_predictor_max_num_epochs", None),
        "predictor_batch_size": args.adaln_predictor_batch_size,
        "predictor_grad_accum_steps": int(getattr(args, "adaln_predictor_grad_accum_steps", 1)),
        "predictor_parallel": bool(getattr(args, "parallel", False)),
        "predictor_lr": args.adaln_predictor_lr,
        "predictor_alpha_lpips": float(getattr(args, "adaln_predictor_alpha_lpips", 0.0)),
        "predictor_alpha_liqe": float(getattr(args, "adaln_predictor_alpha_liqe", 1.0)),
        "predictor_iqa_metric": str(getattr(args, "adaln_predictor_iqa_metric", "liqe")),
        "predictor_alpha_iqa": _resolve_alpha_iqa(args),
        "predictor_alpha_tv": float(getattr(args, "adaln_predictor_alpha_tv", 0.0)),
        "predictor_tv_norm": str(getattr(args, "adaln_predictor_tv_norm", "l1")),
        "predictor_early_stop_patience": int(getattr(args, "adaln_predictor_early_stop_patience", 5)),
        "topk_summary": utils._summarize_topk_map(global_topk_info),
    }
    summary_path = os.path.join(report_dir, "adaln_predictor.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] predictor checkpoint: {predictor_ckpt_path}")
    print(f"[done] summary report: {summary_path}")


if __name__ == "__main__":
    main()
