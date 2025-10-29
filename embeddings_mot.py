# Copyright (c) 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# MOT-specific embedding functions for Video-As-Prompt
# Extracted from Video-As-Prompt modified diffusers

from typing import Optional, Tuple, Union

import torch

# Import get_1d_rotary_pos_embed from official diffusers
from diffusers.models.embeddings import get_1d_rotary_pos_embed


def get_3d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    mot_num: int = 0,  # MOT-specific parameter
    ref_type: str = "continous_negative",  # MOT-specific parameter
    start_point: int = 50,
    gap: int = 30,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure, with MOT support.
    
    This is the MOT-modified version that supports motion transfer by handling
    reference video position embeddings differently.

    Args:
        embed_dim: (`int`):
            The embedding dimension size, corresponding to hidden_size_head.
        crops_coords (`Tuple[int]`):
            The top-left and bottom-right coordinates of the crop.
        grid_size (`Tuple[int]`):
            The grid size of the spatial positional embedding (height, width).
        temporal_size (`int`):
            The size of the temporal dimension.
        theta (`float`):
            Scaling factor for frequency computation.
        grid_type (`str`):
            Whether to use "linspace" or "slice" to compute grids.
        mot_num (`int`):
            Number of motion reference videos (MOT-specific).
        ref_type (`str`):
            Type of reference video position encoding (MOT-specific).

    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`: cos and sin positional embeddings.
    """
    if use_real is not True:
        raise ValueError("`use_real = False` is not currently supported for get_3d_rotary_pos_embed")

    if grid_type == "linspace":
        start, stop = crops_coords
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.linspace(
            start[0], stop[0] * (grid_size_h - 1) / grid_size_h, grid_size_h, device=device, dtype=torch.float32
        )
        grid_w = torch.linspace(
            start[1], stop[1] * (grid_size_w - 1) / grid_size_w, grid_size_w, device=device, dtype=torch.float32
        )
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
        grid_t = torch.linspace(
            0, temporal_size * (temporal_size - 1) / temporal_size, temporal_size, device=device, dtype=torch.float32
        )
        
        # MOT-specific: Handle reference video position embeddings
        if mot_num > 0:
            if ref_type == "continous_negative":
                orig_t_start = 0
                orig_t_stop = temporal_size * (temporal_size - 1) / temporal_size
                
                t_range = orig_t_stop - orig_t_start + 1
                
                temporal_size = temporal_size * mot_num
                grid_t = torch.linspace(-mot_num * t_range, -1, temporal_size, device=device, dtype=torch.float32)

            elif ref_type == "discrete_long_reference":
                start_offsets = start_point + torch.arange(mot_num, device=device, dtype=torch.float32) * gap
                base_range = torch.arange(temporal_size, device=device, dtype=torch.float32)
                grid_t = start_offsets.unsqueeze(1) + base_range
                grid_t = grid_t.flatten().to(device=device, dtype=torch.float32)
            else:
                raise ValueError(f"Invalid {ref_type} passed for `ref_type`.")
                
    elif grid_type == "slice":
        max_h, max_w = max_size
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
        grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
        if mot_num > 0:
            grid_t = torch.arange(-mot_num * temporal_size, 0, device=device, dtype=torch.float32)
    else:
        raise ValueError("Invalid value passed for `grid_type`.")

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta, use_real=True)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

    # BroadCast and concatenate temporal and spatial frequencies (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = freqs_h[None, :, None, :].expand(
            temporal_size, -1, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_h
        freqs_w = freqs_w[None, None, :, :].expand(
            temporal_size, grid_size_h, -1, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_w

        freqs = torch.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(
            temporal_size * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin

