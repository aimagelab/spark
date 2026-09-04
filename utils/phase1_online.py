"""SPARK Phase 1 -- online channel selection via EMA (Algorithm 1).

Dominant channels are identified *online*, while streaming mini-batches of the
optimization set through the frozen backbone, rather than by a full pass over the
dataset:

  * per mini-batch, the per-channel activation magnitude of every block/stream is
    the mean absolute activation over tokens and over the batch (Eq. 1);
  * an exponential moving average smooths it across iterations (Eq. 3),
        m_t = lambda * m_{t-1} + (1 - lambda) * a_t,  m_0 = 0;
  * every W updates the provisional top-K set is snapshotted (Eq. 4);
  * once P consecutive window transitions have a mean Jaccard overlap >= tau for
    every block and stream, the ranking is declared converged and the scan stops
    early (Eq. 5).

m_0 = 0 makes the EMA biased towards zero for small t by the uniform factor
(1 - lambda^t). That factor is identical for every channel of a given block and
stream, so it cannot change their relative order; no bias correction is applied,
matching the algorithm as stated.

The returned structure is the same one the full-pass selector produces, so
Phase 2 consumes it unchanged:

    {block_idx: {"hs_topk_idx": [...], "hs_topk_val": [...],
                 "ehs_topk_idx": [...] | None, "ehs_topk_val": [...] | None}}
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

from utils.adaln_utils import (
    _channel_mean_abs_vector,
    _channel_std_vector,
    _compute_target_size,
    _encode_input,
    _load_rgb_image,
    _prepare_tensor,
    _register_capture_hooks,
    _select_k_from_scores,
)

STREAM_KEYS = ("hs", "ehs")


def _jaccard(a: Set[int], b: Set[int]) -> float:
    """Jaccard overlap of two index sets; two empty sets count as identical."""
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b)) / float(len(union))


def _expand_conditioning(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if int(tensor.shape[0]) == batch_size:
        return tensor
    if int(tensor.shape[0]) == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise RuntimeError(
        f"Cannot expand conditioning tensor from batch {int(tensor.shape[0])} to {batch_size}."
    )


@torch.no_grad()
def _minibatch_channel_scores(
    pixel_values: torch.Tensor,
    vae,
    transformer,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    weight_dtype: torch.dtype,
    importance_mode: str,
) -> Dict[int, Dict[str, Optional[torch.Tensor]]]:
    """a_t for one mini-batch: per-block, per-stream channel magnitude (Eq. 1).

    Only the transformer forward is needed -- Phase 1 never decodes an image and
    never builds a graph, which is what makes it a cheap partial forward pass.
    """
    batch_size = int(pixel_values.shape[0])
    captures: List[Dict[str, Any]] = []
    handles = _register_capture_hooks(transformer, captures, capture_grad=False)
    try:
        model_input = _encode_input(vae, pixel_values, weight_dtype)
        transformer(
            hidden_states=model_input,
            timestep=timesteps,
            encoder_hidden_states=_expand_conditioning(prompt_embeds, batch_size),
            pooled_projections=_expand_conditioning(pooled_prompt_embeds, batch_size),
            return_dict=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    # Both reducers average over every dim but the last (batch and tokens), which
    # is exactly Eq. 1 averaged over the mini-batch.
    score_fn = _channel_std_vector if importance_mode == "std" else _channel_mean_abs_vector

    scores: Dict[int, Dict[str, Optional[torch.Tensor]]] = OrderedDict()
    for act in captures:
        block_idx = int(act["block_index"])
        hidden = act["hidden_states"]
        encoder = act.get("encoder_hidden_states")
        scores[block_idx] = {
            "hs": score_fn(hidden.detach()).to(torch.float32),
            # the final SD3 block has no encoder-stream output
            "ehs": score_fn(encoder.detach()).to(torch.float32) if encoder is not None else None,
        }
    return scores


def _iter_minibatches(
    pairs: Sequence[Dict[str, Optional[str]]],
    batch_size: int,
) -> List[List[Dict[str, Optional[str]]]]:
    return [list(pairs[i : i + batch_size]) for i in range(0, len(pairs), batch_size)]


def select_channels_online_ema(
    pairs: List[Dict[str, Optional[str]]],
    vae,
    transformer,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    weight_dtype: torch.dtype,
    args,
) -> Dict[int, Dict[str, Any]]:
    """Algorithm 1. Returns the selected channel map, keyed by block index."""

    topk = int(args.topk)
    lam = float(getattr(args, "phase1_ema_decay", 0.95))
    window = max(1, int(getattr(args, "phase1_window", 5)))
    transitions = max(1, int(getattr(args, "phase1_window_transitions", 4)))
    tau = float(getattr(args, "phase1_stability_tau", 0.9))
    batch_size = max(1, int(getattr(args, "phase1_batch_size", 16)))
    importance_mode = str(getattr(args, "phase1_importance_mode", "mean_abs"))
    selection_mode = str(getattr(args, "selection_mode", "topk"))

    if not 0.0 <= lam < 1.0:
        raise ValueError(f"--phase1_ema_decay must be in [0, 1), got {lam}")
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"--phase1_stability_tau must be in [0, 1], got {tau}")

    # m_t, per block and stream (line 1: m_0 <- 0, materialized on first sight).
    ema: Dict[int, Dict[str, Optional[torch.Tensor]]] = OrderedDict()
    # Snapshots I_w of the provisional top-K set, most recent last.
    snapshots: Dict[Tuple[int, str], List[Set[int]]] = {}

    batches = _iter_minibatches(pairs, batch_size)
    converged = False
    windows_done = 0
    images_seen = 0
    stop_reason = "exhausted optimization set"

    progress = tqdm(batches, desc="phase1-online-ema", dynamic_ncols=True)
    for step, batch in enumerate(progress, start=1):  # line 3-4: t <- t + 1
        # Images of differing sizes cannot be stacked; group by shape and average
        # the resulting scores so the mini-batch stays one EMA update.
        grouped: Dict[Tuple[int, int], List[torch.Tensor]] = OrderedDict()
        for pair in batch:
            try:
                lr = _load_rgb_image(pair["lr"])
            except Exception as exc:
                raise RuntimeError(f"Failed to load input image '{pair['lr']}': {exc}") from exc
            ori_width, ori_height = lr.size
            new_width, new_height, _ = _compute_target_size(
                ori_width, ori_height, args.upscale, args.process_size
            )
            tensor = _prepare_tensor(lr, (new_height, new_width), args.device, weight_dtype)
            grouped.setdefault((new_height, new_width), []).append(tensor)

        # line 5: forward pass without gradients, collect a_t
        batch_scores: Dict[int, Dict[str, Optional[torch.Tensor]]] = OrderedDict()
        total_in_batch = 0
        for tensors in grouped.values():
            group = torch.cat(tensors, dim=0)
            group_n = int(group.shape[0])
            group_scores = _minibatch_channel_scores(
                group,
                vae,
                transformer,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                weight_dtype,
                importance_mode,
            )
            for block_idx, streams in group_scores.items():
                entry = batch_scores.setdefault(block_idx, {"hs": None, "ehs": None})
                for stream in STREAM_KEYS:
                    value = streams.get(stream)
                    if value is None:
                        continue
                    weighted = value * float(group_n)
                    entry[stream] = weighted if entry[stream] is None else entry[stream] + weighted
            total_in_batch += group_n

        for streams in batch_scores.values():
            for stream in STREAM_KEYS:
                if streams[stream] is not None:
                    streams[stream] = streams[stream] / float(max(1, total_in_batch))

        images_seen += total_in_batch

        # line 6: m_t <- lambda * m_{t-1} + (1 - lambda) * a_t
        for block_idx, streams in batch_scores.items():
            slot = ema.setdefault(block_idx, {"hs": None, "ehs": None})
            for stream in STREAM_KEYS:
                value = streams[stream]
                if value is None:
                    continue
                if slot[stream] is None:
                    slot[stream] = torch.zeros_like(value)
                slot[stream] = lam * slot[stream] + (1.0 - lam) * value

        # line 7: snapshot the ranking every W steps
        if step % window != 0:
            continue

        windows_done += 1  # line 8: w <- w + 1
        for block_idx, streams in ema.items():
            for stream in STREAM_KEYS:
                scores = streams[stream]
                if scores is None:
                    continue
                # line 9: provisional top-K of the current EMA scores
                _, idx = _select_k_from_scores(scores, topk=topk, selection_mode="topk")
                snapshots.setdefault((block_idx, stream), []).append(
                    set(int(i) for i in idx.detach().cpu().tolist())
                )

        # line 10: only meaningful once more than P windows exist
        if windows_done <= transitions:
            continue

        # line 11-12: mean Jaccard over the last P window transitions, all keys
        all_stable = True
        worst = 1.0
        for history in snapshots.values():
            if len(history) < transitions + 1:
                all_stable = False
                break
            recent = history[-(transitions + 1) :]
            mean_j = sum(
                _jaccard(recent[i + 1], recent[i]) for i in range(transitions)
            ) / float(transitions)
            worst = min(worst, mean_j)
            if mean_j < tau:
                all_stable = False

        progress.set_postfix(window=windows_done, worst_jaccard=f"{worst:.3f}")

        if all_stable:  # line 13: ranking has converged
            converged = True
            stop_reason = (
                f"stability criterion met (min mean Jaccard {worst:.3f} >= tau {tau})"
            )
            break

    progress.close()

    if not ema:
        raise RuntimeError("Phase 1 captured no activations; check the optimization set.")

    print(
        f"[phase1-ema] {stop_reason}; images={images_seen} "
        f"mini-batches={min(len(batches), max(1, images_seen // max(1, batch_size)))} "
        f"windows={windows_done} converged={converged}"
    )

    # line 18: freeze the final selection. The stability loop always tracks the
    # top-K set; the ablation modes (bottom-K / random) only change this last
    # step, leaving the rest of the procedure identical.
    info: Dict[int, Dict[str, Any]] = OrderedDict()
    for block_idx, streams in ema.items():
        hs_scores = streams["hs"]
        if hs_scores is None:
            continue
        hs_vals, hs_idx = _select_k_from_scores(
            hs_scores, topk=topk, selection_mode=selection_mode
        )

        ehs_idx_list = None
        ehs_val_list = None
        if streams["ehs"] is not None:
            ehs_vals, ehs_idx = _select_k_from_scores(
                streams["ehs"], topk=topk, selection_mode=selection_mode
            )
            ehs_idx_list = ehs_idx.detach().cpu().tolist()
            ehs_val_list = ehs_vals.detach().cpu().tolist()

        info[block_idx] = {
            "hs_topk_idx": hs_idx.detach().cpu().tolist(),
            "hs_topk_val": hs_vals.detach().cpu().tolist(),
            "ehs_topk_idx": ehs_idx_list,
            "ehs_topk_val": ehs_val_list,
        }

    return info
