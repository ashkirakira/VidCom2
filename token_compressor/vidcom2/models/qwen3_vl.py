from typing import Optional, Union, List, Tuple
import json
import os
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
    is_torchdynamo_compiling,
)

from token_compressor.vidcom2 import (
    select_low_var_channels,
    compute_gaussian_scores,
    compute_local_variation,
    compute_scales,
    select_outlier_indices,
    _map_linear_offset,
)


# Set by lmms_eval/models/chat/qwen3_vl.py before model.generate() to identify
# the current sample. Read here when DUMP_BUDGET=1 to name the output directory.
_current_video_id: Optional[str] = None


def _compute_keep_indices(
    flat_features: Tensor, grid_thw: Tensor, spatial_merge_size: int, base_scale: float
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Runs VidCom2 scoring; returns (kept_indices, scales, raw + 6 norm variants)."""
    t, h, w = grid_thw.tolist()
    frame_tokens = (h * w) // (spatial_merge_size ** 2)
    if frame_tokens <= 0 or flat_features.numel() == 0:
        return torch.arange(flat_features.shape[0], device=flat_features.device), None, None, None, None, None, None, None, None, None

    sel_feat = select_low_var_channels(flat_features)
    vid_score, frame_score = compute_gaussian_scores(sel_feat, frame_tokens)

    local_variation = -compute_local_variation(sel_feat, frame_tokens).squeeze(-1)
    local_variation_norm_l2 = F.normalize(local_variation, p=2, dim=0)
    local_variation_z_score = (local_variation - local_variation.mean()) / (local_variation.std() + 1e-8)
    local_variation_norm_minmax = (local_variation - local_variation.min()) / (local_variation.max() - local_variation.min() + 1e-8)

    global_uniqueness = -vid_score.mean(dim=-1)
    global_uniqueness_norm_l2 = F.normalize(global_uniqueness, p=2, dim=0)
    global_uniqueness_z_score = (global_uniqueness - global_uniqueness.mean()) / (global_uniqueness.std() + 1e-8)
    global_uniqueness_norm_minmax = (global_uniqueness - global_uniqueness.min()) / (global_uniqueness.max() - global_uniqueness.min() + 1e-8)

    combined_tail = (global_uniqueness_norm_l2[1:] + local_variation_norm_l2) / 2
    frame_budgeting = torch.cat([global_uniqueness_norm_l2[0:1], combined_tail])

    scales = compute_scales(frame_budgeting, base_scale)
    indices = select_outlier_indices(vid_score + frame_score, scales, frame_tokens)
    return (
        _map_linear_offset(indices, frame_tokens),
        scales,
        local_variation,
        local_variation_norm_l2,
        local_variation_z_score,
        local_variation_norm_minmax,
        global_uniqueness,
        global_uniqueness_norm_l2,
        global_uniqueness_z_score,
        global_uniqueness_norm_minmax,
    )


def Qwen3VLModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    """Patched forward that enables VidCom2 token compression for Qwen3-VL."""

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # Pre-compute position ids before pruning so we can safely slice them later.
    if position_ids is None:
        attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask.get(
            "full_attention", None
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )

        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    compression_on = (
        os.getenv("COMPRESSOR") == "vidcom2"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            # Keep default behavior for multi-sample batches to avoid shape mismatches.
            compression_on = False

    if compression_on:
        merge_size = self.visual.spatial_merge_size
        base_scale = float(os.getenv("R_RATIO", "0.25"))
        split_sizes = (video_grid_thw.prod(-1) // merge_size**2).tolist()

        video_splits = torch.split(video_embeds, split_sizes)
        deepstack_splits: List[List[Tensor]] = []
        if deepstack_video_embeds is not None:
            deepstack_splits = [list(torch.split(layer, split_sizes)) for layer in deepstack_video_embeds]

        kept_indices: List[Tensor] = []
        kept_video_chunks: List[Tensor] = []
        kept_deepstack: List[List[Tensor]] = [[] for _ in deepstack_splits]
        offset = 0
        dump_enabled = bool(os.getenv("DUMP_BUDGET"))
        dump_scales: List[float] = []
        dump_frame_tokens = 0
        dump_lv_l2: List[float] = []
        dump_lv_z: List[float] = []
        dump_lv_mm: List[float] = []
        dump_gu_l2: List[float] = []
        dump_gu_z: List[float] = []
        dump_gu_mm: List[float] = []
        dump_lv_raw: List[float] = []
        dump_gu_raw: List[float] = []

        for grid, feat in zip(video_grid_thw, video_splits):
            (
                keep_local, seg_scales,
                lv_raw, lv_l2, lv_z, lv_mm,
                gu_raw, gu_l2, gu_z, gu_mm,
            ) = _compute_keep_indices(
                flat_features=feat,
                grid_thw=grid,
                spatial_merge_size=merge_size,
                base_scale=base_scale,
            )
            kept_indices.append(keep_local + offset)
            kept_video_chunks.append(feat[keep_local])
            for layer_idx, split_list in enumerate(deepstack_splits):
                kept_deepstack[layer_idx].append(split_list.pop(0)[keep_local])
            offset += feat.shape[0]

            if dump_enabled and seg_scales is not None:
                dump_scales.extend(seg_scales.cpu().tolist())
                dump_lv_raw.extend(lv_raw.cpu().tolist())
                dump_lv_l2.extend(lv_l2.cpu().tolist())
                dump_lv_z.extend(lv_z.cpu().tolist())
                dump_lv_mm.extend(lv_mm.cpu().tolist())
                dump_gu_raw.extend(gu_raw.cpu().tolist())
                dump_gu_l2.extend(gu_l2.cpu().tolist())
                dump_gu_z.extend(gu_z.cpu().tolist())
                dump_gu_mm.extend(gu_mm.cpu().tolist())
                h, w = grid[1].item(), grid[2].item()
                dump_frame_tokens = (h * w) // (merge_size ** 2)

        if dump_enabled and _current_video_id and dump_scales:
            dump_dir = Path(os.getenv("DUMP_BUDGET_DIR", "./norm_data")) / _current_video_id
            dump_dir.mkdir(parents=True, exist_ok=True)
            ks = [max(1, round(s * dump_frame_tokens)) for s in dump_scales]
            with (dump_dir / "budget.json").open("w") as f:
                json.dump({
                    "scales": dump_scales,
                    "ks": ks,
                    "frame_tokens": dump_frame_tokens,
                    "r_ratio": base_scale,
                    "local_variation": dump_lv_raw,
                    "local_variation_norm_l2": dump_lv_l2,
                    "local_variation_z_score": dump_lv_z,
                    "local_variation_norm_minmax": dump_lv_mm,
                    "global_uniqueness": dump_gu_raw,
                    "global_uniqueness_norm_l2": dump_gu_l2,
                    "global_uniqueness_z_score": dump_gu_z,
                    "global_uniqueness_norm_minmax": dump_gu_mm,
                }, f, indent=2)

        kept_indices = torch.sort(torch.cat(kept_indices)).values
        video_embeds = torch.cat(kept_video_chunks, dim=0)
        if kept_deepstack:
            deepstack_video_embeds = [torch.cat(chunks, dim=0) for chunks in kept_deepstack]

        video_token_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        kept_video_positions = video_token_positions[kept_indices]
        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_video_positions = all_positions[~video_mask[..., 0][0]]
        keep_token_indices = torch.cat((non_video_positions, kept_video_positions)).sort().values

        def _prune_attention(attn: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if attn is None:
                return None
            if attn.dim() == 2:
                return attn[:, keep_token_indices]
            if attn.dim() == 4:
                return attn[:, :, keep_token_indices, :][:, :, :, keep_token_indices]
            return attn

        inputs_embeds = inputs_embeds[:, keep_token_indices, :]
        if input_ids is not None:
            input_ids = input_ids[:, keep_token_indices]
        attention_mask = (
            {k: _prune_attention(v) for k, v in attention_mask.items()}
            if isinstance(attention_mask, dict)
            else _prune_attention(attention_mask)
        )
        position_ids = position_ids[:, :, keep_token_indices]

        if image_mask is not None:
            image_mask = image_mask[:, keep_token_indices, :]
        if video_mask is not None:
            video_mask = video_mask[:, keep_token_indices, :]

        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.dtype))

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask_compact = image_mask[..., 0]
        video_mask_compact = video_mask[..., 0]
        visual_pos_masks = image_mask_compact | video_mask_compact
        deepstack_visual_embeds = []
        image_mask_joint = image_mask_compact[visual_pos_masks]
        video_mask_joint = video_mask_compact[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask_compact = image_mask[..., 0]
        visual_pos_masks = image_mask_compact
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask_compact = video_mask[..., 0]
        visual_pos_masks = video_mask_compact
        deepstack_visual_embeds = deepstack_video_embeds

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )
