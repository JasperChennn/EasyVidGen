# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
except ImportError:  # pragma: no cover
    BlockMask = Any  # type: ignore
    create_block_mask = None  # type: ignore
    flex_attention = None  # type: ignore

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _flex_attn_causal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: "BlockMask",
) -> torch.Tensor:
    """Block-wise causal self-attention via flex_attention (query/key/value: [B, L, H, D])."""
    if flex_attention is None:
        raise ImportError(
            "Causal block attention requires PyTorch with torch.nn.attention.flex_attention (e.g. PyTorch 2.5+)."
        )
    padded_length = math.ceil(query.shape[1] / 128) * 128 - query.shape[1]
    if padded_length > 0:
        pad_shape_q = (query.shape[0], padded_length, query.shape[2], query.shape[3])
        pad_shape_kv = (key.shape[0], padded_length, key.shape[2], key.shape[3])
        zq = query.new_zeros(pad_shape_q)
        zk = key.new_zeros(pad_shape_kv)
        zv = value.new_zeros(pad_shape_kv)
        roped_query = torch.cat([query, zq], dim=1)
        roped_key = torch.cat([key, zk], dim=1)
        padded_v = torch.cat([value, zv], dim=1)
    else:
        roped_query, roped_key, padded_v = query, key, value

    out = flex_attention(
        query=roped_query.transpose(1, 2),
        key=roped_key.transpose(1, 2),
        value=padded_v.transpose(1, 2),
        block_mask=block_mask,
    )
    if padded_length > 0:
        out = out[:, :, :-padded_length]
    return out.transpose(1, 2)


def _prepare_blockwise_causal_attn_mask(
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int = 1,
    local_attn_size: int = -1,
) -> "BlockMask":
    """Block-wise causal mask: each latent frame block attends to all tokens up to the end of the current chunk."""
    if create_block_mask is None:
        raise ImportError("create_block_mask is not available; upgrade PyTorch.")

    total_length = num_frames * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128 - total_length
    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    frame_indices = torch.arange(
        0,
        total_length,
        step=frame_seqlen * num_frame_per_block,
        device=device,
        dtype=torch.long,
    )
    for tmp in frame_indices.tolist():
        ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

    def attention_mask(b, h, q_idx, kv_idx):
        if local_attn_size == -1:
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.debug(
            "Using block-wise causal flex mask: num_frames=%s frame_seqlen=%s block=%s frames local_attn=%s",
            num_frames,
            frame_seqlen,
            num_frame_per_block,
            local_attn_size,
        )
    return block_mask


def _prepare_blockwise_causal_attn_mask_i2v(
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int = 4,
    local_attn_size: int = -1,
) -> "BlockMask":
    """I2V: first latent frame is its own block; remaining frames use ``num_frame_per_block`` (see causal_transformer)."""
    if create_block_mask is None:
        raise ImportError("create_block_mask is not available; upgrade PyTorch.")

    total_length = num_frames * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128 - total_length
    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    ends[:frame_seqlen] = frame_seqlen
    frame_indices = torch.arange(
        frame_seqlen,
        total_length,
        step=frame_seqlen * num_frame_per_block,
        device=device,
        dtype=torch.long,
    )
    for tmp in frame_indices.tolist():
        ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

    def attention_mask(b, h, q_idx, kv_idx):
        if local_attn_size == -1:
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.debug(
            "Using I2V block-wise causal flex mask: num_frames=%s frame_seqlen=%s block=%s frames local_attn=%s",
            num_frames,
            frame_seqlen,
            num_frame_per_block,
            local_attn_size,
        )
    return block_mask


def _prepare_teacher_forcing_mask(
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int = 1,
) -> "BlockMask":
    """Teacher forcing: sequence is ``[clean tokens | noisy tokens]`` of length ``2 * num_frames * frame_seqlen``."""
    if create_block_mask is None:
        raise ImportError("create_block_mask is not available; upgrade PyTorch.")

    total_length = num_frames * frame_seqlen * 2
    padded_length = math.ceil(total_length / 128) * 128 - total_length
    clean_ends = num_frames * frame_seqlen

    context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    attention_block_size = frame_seqlen * num_frame_per_block
    frame_indices = torch.arange(
        0,
        num_frames * frame_seqlen,
        step=attention_block_size,
        device=device,
        dtype=torch.long,
    )
    for start in frame_indices.tolist():
        context_ends[start : start + attention_block_size] = start + attention_block_size

    noisy_image_start_list = torch.arange(
        num_frames * frame_seqlen,
        total_length,
        step=attention_block_size,
        device=device,
        dtype=torch.long,
    )
    noisy_image_end_list = noisy_image_start_list + attention_block_size

    for block_index, (start, end) in enumerate(zip(noisy_image_start_list.tolist(), noisy_image_end_list.tolist())):
        noise_noise_starts[start:end] = start
        noise_noise_ends[start:end] = end
        noise_context_ends[start:end] = block_index * attention_block_size

    def attention_mask(b, h, q_idx, kv_idx):
        clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
        c1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
        c2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
        noise_mask = (q_idx >= clean_ends) & (c1 | c2)
        eye_mask = q_idx == kv_idx
        return eye_mask | clean_mask | noise_mask

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.debug(
            "Using teacher-forcing flex mask: num_frames=%s frame_seqlen=%s block=%s frames",
            num_frames,
            frame_seqlen,
            num_frame_per_block,
        )
    return block_mask


def _sdpa_attention_cached(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Scaled dot-product attention; query/key/value layout ``[B, L, H, D]`` (same as causal Wan ``attention``)."""
    # [B, H, L, D]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2)


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_mask: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Self-attn is called with ``encoder_hidden_states is None``; cross-attn passes text/image as K/V (no RoPE).
        is_self_attention = encoder_hidden_states is None
        cross_attn_kv_cache = kwargs.get("cross_attn_kv_cache")
        cross_attn_cache_ready = (
            not is_self_attention
            and cross_attn_kv_cache is not None
            and cross_attn_kv_cache.get("filled", False)
        )

        def apply_rotary_emb(
            hidden_states: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
        ):
            x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
            cos = freqs_cos[..., 0::2]
            sin = freqs_sin[..., 1::2]
            out = torch.empty_like(hidden_states)
            out[..., 0::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos
            return out.type_as(hidden_states)

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None and not cross_attn_cache_ready:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Cross-attn: reuse cached encoder K/V (``causal_transformer`` crossattn_cache); only Q from current hidden states.
        if cross_attn_cache_ready:
            query = attn.to_q(hidden_states)
            query = attn.norm_q(query)
            query = query.unflatten(2, (attn.heads, -1))
            key = cross_attn_kv_cache["k"]
            value = cross_attn_kv_cache["v"]
            if rotary_emb is not None:
                query = apply_rotary_emb(query, *rotary_emb)
                key = apply_rotary_emb(key, *rotary_emb)

            hidden_states_img = None
            if "k_img" in cross_attn_kv_cache:
                key_img = cross_attn_kv_cache["k_img"]
                value_img = cross_attn_kv_cache["v_img"]
                hidden_states_img = dispatch_attention_fn(
                    query,
                    key_img,
                    value_img,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
                hidden_states_img = hidden_states_img.flatten(2, 3)
                hidden_states_img = hidden_states_img.type_as(query)

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.type_as(query)
            if hidden_states_img is not None:
                hidden_states = hidden_states + hidden_states_img
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Cross-attention: cache encoder K/V on first forward (``causal_transformer`` ``crossattn_cache``).
        if not is_self_attention and cross_attn_kv_cache is not None and not cross_attn_kv_cache.get("filled", False):
            cross_attn_kv_cache["k"] = key.clone()
            cross_attn_kv_cache["v"] = value.clone()
            if encoder_hidden_states_img is None:
                cross_attn_kv_cache["filled"] = True

        # Per-layer ``kv_cache`` dict is only for **self-attention** (causal KV over video tokens).
        kv_cache_dict = kwargs.get("kv_cache")
        if kv_cache_dict is not None and is_self_attention:
            if encoder_hidden_states_img is not None:
                raise ValueError("kv_cache self-attention path does not support add_k_proj / image tokens.")
            if block_mask is not None:
                raise ValueError("Use either kv_cache inference or block_mask, not both.")
            if rotary_emb is None:
                raise ValueError(
                    "Self-attention with ``kv_cache`` requires ``rotary_emb`` from "
                    "``WanRotaryPosEmbed.forward_causal`` (cross-attention does not use this path)."
                )
            frame_seqlen = kwargs.get("kv_cache_frame_seqlen")
            if frame_seqlen is None:
                raise ValueError("kv_cache_frame_seqlen is required when kv_cache is set.")
            current_start = int(kwargs.get("kv_cache_current_start", 0))
            cache_start = kwargs.get("kv_cache_cache_start")
            if cache_start is None:
                cache_start = current_start
            else:
                cache_start = int(cache_start)
            local_attn_size = int(kwargs.get("kv_cache_local_attn_size", -1))
            sink_size = int(kwargs.get("kv_cache_sink_size", 0))

            roped_query = apply_rotary_emb(query, *rotary_emb).type_as(value)
            roped_key = apply_rotary_emb(key, *rotary_emb).type_as(value)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = sink_size * frame_seqlen
            kv_cache_size = kv_cache_dict["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * frame_seqlen

            if (
                local_attn_size != -1
                and (current_end > kv_cache_dict["global_end_index"].item())
                and (num_new_tokens + kv_cache_dict["local_end_index"].item() > kv_cache_size)
            ):
                num_evicted_tokens = num_new_tokens + kv_cache_dict["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache_dict["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache_dict["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache_dict["k"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                kv_cache_dict["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache_dict["v"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                local_end_index = (
                    kv_cache_dict["local_end_index"].item()
                    + current_end
                    - kv_cache_dict["global_end_index"].item()
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache_dict["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache_dict["v"][:, local_start_index:local_end_index] = value
            else:
                local_end_index = (
                    kv_cache_dict["local_end_index"].item()
                    + current_end
                    - kv_cache_dict["global_end_index"].item()
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache_dict["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache_dict["v"][:, local_start_index:local_end_index] = value

            k_slice = kv_cache_dict["k"][:, max(0, local_end_index - max_attention_size) : local_end_index]
            v_slice = kv_cache_dict["v"][:, max(0, local_end_index - max_attention_size) : local_end_index]
            hidden_states = _sdpa_attention_cached(roped_query, k_slice, v_slice)
            kv_cache_dict["global_end_index"].fill_(current_end)
            kv_cache_dict["local_end_index"].fill_(local_end_index)

            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.type_as(query)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

        is_tf = (
            kwargs.get("teacher_forcing", False)
            and block_mask is not None
            and is_self_attention
        )
        if rotary_emb is not None:
            if is_tf:
                l_half = query.shape[1] // 2
                cos, sin = rotary_emb[0], rotary_emb[1]
                q1 = apply_rotary_emb(query[:, :l_half], cos[:, :l_half], sin[:, :l_half])
                q2 = apply_rotary_emb(query[:, l_half:], cos[:, l_half:], sin[:, l_half:])
                k1 = apply_rotary_emb(key[:, :l_half], cos[:, :l_half], sin[:, :l_half])
                k2 = apply_rotary_emb(key[:, l_half:], cos[:, l_half:], sin[:, l_half:])
                query = torch.cat([q1, q2], dim=1)
                key = torch.cat([k1, k2], dim=1)
            else:
                query = apply_rotary_emb(query, *rotary_emb)
                key = apply_rotary_emb(key, *rotary_emb)

        # Block-wise causal self-attention (flex_attention); mutually exclusive with added image KV in this path.
        if block_mask is not None:
            if encoder_hidden_states_img is not None:
                raise ValueError("block_mask causal path does not support add_k_proj / image tokens in the same attention.")
            hidden_states = _flex_attn_causal(query, key, value, block_mask)
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.type_as(query)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            if cross_attn_kv_cache is not None and not cross_attn_kv_cache.get("filled", False):
                cross_attn_kv_cache["k_img"] = key_img.clone()
                cross_attn_kv_cache["v_img"] = value_img.clone()
                cross_attn_kv_cache["filled"] = True

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "The WanAttnProcessor2_0 class is deprecated and will be removed in a future version. "
            "Please use WanAttnProcessor instead. "
        )
        deprecate("WanAttnProcessor2_0", "1.0.0", deprecation_message, standard_warn=False)
        return WanAttnProcessor(*args, **kwargs)


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin

    def forward_causal(
        self,
        post_patch_height: int,
        post_patch_width: int,
        start_frame: int,
        num_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RoPE cos/sin for a contiguous range of latent frames (for KV-cache inference).

        Token order matches :meth:`forward` (flatten F then H then W). Temporal frequencies use
        ``freqs_*[start_frame : start_frame + num_frames]``, aligned with ``causal_transformer.causal_rope_apply``.
        """
        split_sizes = [self.t_dim, self.h_dim, self.w_dim]
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        f_end = start_frame + num_frames
        fc0 = freqs_cos[0][start_frame:f_end]
        fs0 = freqs_sin[0][start_frame:f_end]
        nf = num_frames
        pph, ppw = post_patch_height, post_patch_width

        freqs_cos_f = fc0.view(nf, 1, 1, -1).expand(nf, pph, ppw, -1)
        freqs_sin_f = fs0.view(nf, 1, 1, -1).expand(nf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(nf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(nf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(nf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(nf, pph, ppw, -1)

        freqs_cos_o = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, nf * pph * ppw, 1, -1)
        freqs_sin_o = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, nf * pph * ppw, 1, -1)
        return freqs_cos_o, freqs_sin_o


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        block_mask: Optional[Any] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        current_start: int = 0,
        cache_start: Optional[int] = None,
        frame_seqlen: Optional[int] = None,
        kv_cache_local_attn_size: int = -1,
        kv_cache_sink_size: int = 0,
        cross_attn_kv_cache: Optional[Dict[str, Any]] = None,
        teacher_forcing: bool = False,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            None,
            None,
            rotary_emb,
            block_mask=block_mask,
            kv_cache=kv_cache,
            kv_cache_current_start=current_start,
            kv_cache_cache_start=cache_start,
            kv_cache_frame_seqlen=frame_seqlen,
            kv_cache_local_attn_size=kv_cache_local_attn_size,
            kv_cache_sink_size=kv_cache_sink_size,
            teacher_forcing=teacher_forcing,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention (optional encoder K/V cache; see ``WanAttnProcessor`` / ``causal_transformer`` crossattn_cache)
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            None,
            None,
            cross_attn_kv_cache=cross_attn_kv_cache,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        use_causal_block_attention (`bool`, defaults to `False`):
            If True, self-attention uses block-wise temporal causal masks with ``flex_attention`` (same idea as
            ``tmp/causal_transformer.py``). Requires a recent PyTorch with ``torch.nn.attention.flex_attention``.
        local_attn_size (`int`, defaults to `-1`):
            Sliding window over past frames in units of frames; `-1` means full history within the causal prefix.
        num_frame_per_block (`int`, defaults to `1`):
            Number of latent frames per causal block (uniform schedule) or block size after the first frame (I2V split).
        independent_first_frame (`bool`, defaults to `False`):
            If True and ``image_dim`` is set (I2V), use the first-frame-separated block mask (see ``_prepare_blockwise_causal_attn_mask_i2v``).
        sink_size (`int`, defaults to `0`):
            When using per-layer KV cache (``forward(..., kv_cache=...)``), number of **latent frames** kept as attention
            sink when rolling the cache (same as ``causal_transformer.CausalWanSelfAttention``).
        cross_attn_kv_cache (``forward`` only):
            Optional list (length ``num_layers``) of per-layer dicts from ``init_cross_attn_kv_cache``; caches
            encoder **K/V** for cross-attention so autoregressive steps skip ``to_k``/``to_v`` on text (and image K/V for I2V).
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*": {
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        use_causal_block_attention: bool = False,
        local_attn_size: int = -1,
        num_frame_per_block: int = 1,
        independent_first_frame: bool = False,
        sink_size: int = 0,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
        self._causal_block_mask_cache: Optional[Any] = None
        self._causal_block_mask_key: Optional[Tuple[Any, ...]] = None

    @staticmethod
    def init_kv_cache(
        num_layers: int,
        batch_size: int,
        max_sequence_length: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """Allocate per-layer KV caches (``causal_transformer`` layout: ``k``, ``v``, ``global_end_index``, ``local_end_index``)."""
        caches = []
        for _ in range(num_layers):
            caches.append(
                {
                    "k": torch.zeros(
                        batch_size, max_sequence_length, num_heads, head_dim, dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        batch_size, max_sequence_length, num_heads, head_dim, dtype=dtype, device=device
                    ),
                    "global_end_index": torch.tensor(0, dtype=torch.long, device=device),
                    "local_end_index": torch.tensor(0, dtype=torch.long, device=device),
                }
            )
        return caches

    @staticmethod
    def init_cross_attn_kv_cache(num_layers: int) -> List[Dict[str, Any]]:
        """Per-layer dicts for cross-attention encoder K/V (``filled``, ``k``, ``v``, optional ``k_img`` / ``v_img``)."""
        return [{"filled": False} for _ in range(num_layers)]

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        current_start: int = 0,
        cache_start: Optional[int] = None,
        cross_attn_kv_cache: Optional[List[Dict[str, Any]]] = None,
        clean_hidden_states: Optional[torch.Tensor] = None,
        aug_timestep: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        frame_seqlen = post_patch_height * post_patch_width

        noisy_latents = hidden_states
        teacher_forcing = clean_hidden_states is not None
        if teacher_forcing:
            if kv_cache is not None:
                raise ValueError("Teacher forcing (clean_hidden_states) is incompatible with kv_cache.")
            if not self.config.use_causal_block_attention:
                raise ValueError("Teacher forcing requires use_causal_block_attention=True.")
            if encoder_hidden_states_image is not None:
                raise NotImplementedError(
                    "Teacher forcing with encoder_hidden_states_image (I2V) is not supported."
                )
            if clean_hidden_states.shape != noisy_latents.shape:
                raise ValueError(
                    "clean_hidden_states must match hidden_states shape; "
                    f"got {tuple(clean_hidden_states.shape)} vs {tuple(noisy_latents.shape)}."
                )

        if kv_cache is not None:
            start_frame = current_start // max(frame_seqlen, 1)
            rotary_emb = self.rope.forward_causal(
                post_patch_height,
                post_patch_width,
                start_frame,
                post_patch_num_frames,
            )
            rotary_emb = (rotary_emb[0].to(device=hidden_states.device), rotary_emb[1].to(device=hidden_states.device))
        elif teacher_forcing:
            rc = self.rope(clean_hidden_states)
            rn = self.rope(noisy_latents)
            rotary_emb = (
                torch.cat([rc[0], rn[0]], dim=1),
                torch.cat([rc[1], rn[1]], dim=1),
            )
        else:
            rotary_emb = self.rope(hidden_states)

        if teacher_forcing:
            hidden_states = torch.cat(
                [
                    self.patch_embedding(clean_hidden_states).flatten(2).transpose(1, 2),
                    self.patch_embedding(noisy_latents).flatten(2).transpose(1, 2),
                ],
                dim=1,
            )
        else:
            hidden_states = self.patch_embedding(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep: batch, or batch x seq (wan 2.2 ti2v); teacher forcing uses [aug | noisy] along seq
        if teacher_forcing:
            if timestep.ndim == 1:
                noisy_t = timestep[:, None].expand(-1, post_patch_num_frames)
            else:
                noisy_t = timestep
                if noisy_t.shape[1] != post_patch_num_frames:
                    raise ValueError(
                        "When using teacher forcing, timestep must have length post_patch_num_frames per batch "
                        f"when 2D; got {noisy_t.shape[1]} vs {post_patch_num_frames}."
                    )
            if aug_timestep is None:
                aug_t = torch.zeros_like(noisy_t)
            else:
                if aug_timestep.ndim == 1:
                    aug_t = aug_timestep[:, None].expand(-1, post_patch_num_frames)
                else:
                    aug_t = aug_timestep
                if aug_t.shape != noisy_t.shape:
                    raise ValueError(
                        "aug_timestep must match expanded timestep shape when using teacher forcing; "
                        f"got {aug_t.shape} vs {noisy_t.shape}."
                    )
            ts_seq_len = 2 * post_patch_num_frames
            timestep = torch.cat([aug_t, noisy_t], dim=1).flatten()
        elif timestep.ndim == 2 and timestep.shape[1] == post_patch_num_frames:
            ts_seq_len = post_patch_num_frames * frame_seqlen
            timestep = timestep.unsqueeze(-1).expand(batch_size, -1, frame_seqlen)
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        block_mask: Optional[Any] = None
        if self.config.use_causal_block_attention and kv_cache is None:
            if create_block_mask is None:
                raise ImportError(
                    "use_causal_block_attention requires PyTorch with flex_attention (e.g. 2.5+). "
                    "Set use_causal_block_attention=False or upgrade PyTorch."
                )
            cache_key = (
                hidden_states.device,
                post_patch_num_frames,
                frame_seqlen,
                self.config.num_frame_per_block,
                self.config.local_attn_size,
                self.config.independent_first_frame,
                self.config.image_dim is not None,
                teacher_forcing,
            )
            if self._causal_block_mask_key != cache_key or self._causal_block_mask_cache is None:
                if teacher_forcing:
                    self._causal_block_mask_cache = _prepare_teacher_forcing_mask(
                        hidden_states.device,
                        num_frames=post_patch_num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.config.num_frame_per_block,
                    )
                elif self.config.image_dim is not None and self.config.independent_first_frame:
                    self._causal_block_mask_cache = _prepare_blockwise_causal_attn_mask_i2v(
                        hidden_states.device,
                        num_frames=post_patch_num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.config.num_frame_per_block,
                        local_attn_size=self.config.local_attn_size,
                    )
                else:
                    self._causal_block_mask_cache = _prepare_blockwise_causal_attn_mask(
                        hidden_states.device,
                        num_frames=post_patch_num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.config.num_frame_per_block,
                        local_attn_size=self.config.local_attn_size,
                    )
                self._causal_block_mask_key = cache_key
            block_mask = self._causal_block_mask_cache

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            if kv_cache is not None:
                raise NotImplementedError("KV cache inference is not supported with gradient checkpointing.")
            if cross_attn_kv_cache is not None:
                raise NotImplementedError(
                    "Cross-attention encoder KV cache is not supported with gradient checkpointing."
                )
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    block_mask,
                    teacher_forcing,
                )
        else:
            for i, block in enumerate(self.blocks):
                layer_kv = kv_cache[i] if kv_cache is not None else None
                layer_cross = cross_attn_kv_cache[i] if cross_attn_kv_cache is not None else None
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    block_mask=block_mask,
                    kv_cache=layer_kv,
                    current_start=current_start,
                    cache_start=cache_start,
                    frame_seqlen=frame_seqlen,
                    kv_cache_local_attn_size=self.config.local_attn_size,
                    kv_cache_sink_size=self.config.sink_size,
                    cross_attn_kv_cache=layer_cross,
                    teacher_forcing=teacher_forcing,
                )

        if teacher_forcing:
            hidden_states = hidden_states[:, hidden_states.shape[1] // 2 :]

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            if teacher_forcing:
                temb = temb[:, post_patch_num_frames:, :]
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)