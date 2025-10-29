# Copyright (c) 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# MOT (Motion Transfer) attention processor for Video-As-Prompt
# Extracted from Video-As-Prompt modified diffusers

from typing import Optional

import torch
import torch.nn.functional as F


class CogVideoXAttnMOTProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model with MOT support.
    It applies a rotary embedding on query and key vectors, but does not include spatial normalization.
    
    This processor handles motion transfer by processing reference video attention separately.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnMOTProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,  # Attention module from diffusers.models.attention
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # MOT specific parameters
        is_before_attn: bool = False,
        is_ref_video: Optional[bool] = False,
        text_seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply attention with MOT support.
        
        Args:
            attn: The Attention module
            hidden_states: Input hidden states
            encoder_hidden_states: Encoder hidden states (text embeddings)
            attention_mask: Attention mask
            image_rotary_emb: Rotary position embeddings for images
            is_before_attn: If True, only compute Q, K, V projections (before attention)
            is_ref_video: Whether this is processing reference video
            text_seq_length: Length of text sequence for splitting
            
        Returns:
            If is_before_attn=True: (query, key, value, attention_mask)
            If is_before_attn=False: (hidden_states, encoder_hidden_states)
        """
        if is_before_attn:
            # Phase 1: Compute Q, K, V projections
            text_seq_length = encoder_hidden_states.size(1)

            # Concatenate text and video sequences
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            batch_size, sequence_length, _ = hidden_states.shape

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            # Linear projections
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            # Reshape for multi-head attention
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Apply normalization if available
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # Apply RoPE (Rotary Position Embedding) if needed
            if image_rotary_emb is not None:
                # Import here to avoid circular dependency
                from diffusers.models.embeddings import apply_rotary_emb

                # Apply RoPE only to video tokens (skip text tokens)
                query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                if not attn.is_cross_attention:
                    key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
            
            return query, key, value, attention_mask
        
        else:
            # Phase 2: Post-attention processing
            batch_size, _, sequence_length, head_dim = hidden_states.shape
            
            # Reshape back from multi-head format
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, attn.heads * head_dim)

            # Linear projection
            hidden_states = attn.to_out[0](hidden_states)
            # Dropout
            hidden_states = attn.to_out[1](hidden_states)

            # Split back into text and video sequences
            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
            
            return hidden_states, encoder_hidden_states

