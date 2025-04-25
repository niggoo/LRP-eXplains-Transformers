# File: lxt/efficient/models/roberta.py

import torch
import torch.nn as nn
from functools import partial
import math

# Import LXT rules and core patching utilities
from lxt.efficient.rules import identity_rule_implicit, stop_gradient, divide_gradient
from lxt.efficient.patches import (
    patch_method,
    # Removed patch_attention, patch_cp_attention as they are RoBERTa-incompatible
    dropout_forward,
)

# Import RoBERTa classes and module file (though we won't patch the module directly now)
from transformers.models.roberta import modeling_roberta
from transformers.models.roberta.modeling_roberta import (
    RobertaIntermediate,
    RobertaSelfAttention,     # Target for eager attention patch
    RobertaSdpaSelfAttention, # Target for SDPA attention patch
    # We need the base modeling_roberta for potential other uses if any, but not for attention patching itself
)
from torch.nn import Dropout, LayerNorm # Need LayerNorm to patch it globally

# --- Define Custom Patching Functions for RoBERTa ---

# --- LayerNorm Patch (Unchanged) ---
def roberta_layernorm_forward_lxt(self: nn.LayerNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states_float = hidden_states.float()
    mean = hidden_states_float.mean(-1, keepdim=True)
    variance = hidden_states_float.var(-1, keepdim=True, unbiased=False)
    variance_sg = stop_gradient(variance) # LXT Rule
    hidden_states_normalized = (hidden_states_float - mean) / torch.sqrt(variance_sg + self.eps)
    if self.elementwise_affine:
        hidden_states_normalized = hidden_states_normalized * self.weight.float() + self.bias.float()
    return hidden_states_normalized.to(input_dtype)

cp_roberta_layernorm_forward_lxt = roberta_layernorm_forward_lxt

# --- Intermediate MLP Patch (Unchanged) ---
def roberta_intermediate_forward_lxt(self: RobertaIntermediate, hidden_states: torch.Tensor) -> torch.Tensor:
    dense_output = self.dense(hidden_states)
    activated_output = identity_rule_implicit(self.intermediate_act_fn, dense_output) # LXT Rule
    return activated_output

cp_roberta_intermediate_forward_lxt = roberta_intermediate_forward_lxt


# --- Custom Eager Attention Patch (AttnLRP) ---
def roberta_self_attention_forward_lxt(
    self: RobertaSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask = None,
    head_mask = None,
    encoder_hidden_states = None,
    encoder_attention_mask = None,
    past_key_value = None,
    output_attentions: bool = False,
) -> tuple:
    """Patched RobertaSelfAttention forward for AttnLRP (uniform rule)."""

    # --- Start: Original RobertaSelfAttention Logic (with modifications) ---
    mixed_query_layer = self.query(hidden_states)
    is_cross_attention = encoder_hidden_states is not None
    # Key/Value layer calculation (remains the same logic)
    if is_cross_attention and past_key_value is not None:
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    query_layer = self.transpose_for_scores(mixed_query_layer)

    if self.is_decoder:
        past_key_value = (key_layer, value_layer)

    # --- LXT Rule: Uniform Rule for QK Matmul (Applied to output) ---
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = divide_gradient(attention_scores, 2) # Apply rule here
    # --- End LXT Rule ---

    # Relative position embedding logic (remains the same)
    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        # ... (original relative position logic) ...
        # This part is complex and might require careful handling if gradients through embeddings matter.
        # Assuming LXT focuses on the main matmuls for now.
        seq_len = query_layer.shape[2]
        position_ids_l = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(key_layer.shape[2], dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = positional_embedding.to(dtype=query_layer.dtype)
        if self.position_embedding_type == "relative_key":
             relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
             attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
             relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
             relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
             attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key


    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # Dropout should be handled by the global Dropout patch if self.dropout is nn.Dropout
    # If LXT requires dropout=0 always during LRP, ensure this happens.
    # The global patch should make self.dropout(attention_probs) behave correctly (identity or LRP-compatible)
    attention_probs_after_dropout = self.dropout(attention_probs)

    if head_mask is not None:
        attention_probs_after_dropout = attention_probs_after_dropout * head_mask

    # --- LXT Rule: Uniform Rule for Attention @ Value Matmul (Applied to output) ---
    context_layer = torch.matmul(attention_probs_after_dropout, value_layer)
    context_layer = divide_gradient(context_layer, 2) # Apply rule here
    # --- End LXT Rule ---

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs
    # --- End: Original RobertaSelfAttention Logic ---


# --- Custom SDPA Attention Patch (AttnLRP) ---
def roberta_sdpa_attention_forward_lxt(
    self: RobertaSdpaSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask = None,
    head_mask = None,
    encoder_hidden_states = None,
    encoder_attention_mask = None,
    past_key_value = None,
    output_attentions: bool = False,
) -> tuple:
    """Patched RobertaSdpaSelfAttention forward for AttnLRP (uniform rule)."""

    # Fallback logic (remains the same) - Check if SDPA can be used
    if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
         # Use the *patched* eager forward if SDPA cannot be used
        return roberta_self_attention_forward_lxt(
            self, hidden_states, attention_mask, head_mask, encoder_hidden_states,
            encoder_attention_mask, past_key_value, output_attentions
        )

    # --- Start: Original RobertaSdpaSelfAttention Logic (with modifications) ---
    bsz, tgt_len, _ = hidden_states.size()
    query_layer = self.transpose_for_scores(self.query(hidden_states))
    # Key/Value layer calculation (remains the same logic)
    is_cross_attention = encoder_hidden_states is not None
    current_states = encoder_hidden_states if is_cross_attention else hidden_states
    attention_mask_sdpa = encoder_attention_mask if is_cross_attention else attention_mask # Use correct mask for SDPA

    if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
        key_layer, value_layer = past_key_value
    else:
        key_layer = self.transpose_for_scores(self.key(current_states))
        value_layer = self.transpose_for_scores(self.value(current_states))
        if past_key_value is not None and not is_cross_attention:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

    if self.is_decoder:
        past_key_value = (key_layer, value_layer)

    # Contiguous check (remains the same)
    if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask_sdpa is not None:
         query_layer = query_layer.contiguous()
         key_layer = key_layer.contiguous()
         value_layer = value_layer.contiguous()

    # --- LXT Rules for SDPA ---
    # Apply rule for QK Matmul (before SDPA call)
    query_layer_lxt = divide_gradient(query_layer, 2)
    key_layer_lxt = divide_gradient(key_layer, 2)

    # SDPA call (set dropout_p=0.0)
    is_causal = True if self.is_decoder and not is_cross_attention and attention_mask_sdpa is None and tgt_len > 1 else False
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_layer_lxt, # Use modified Q
        key_layer_lxt,   # Use modified K
        value_layer,     # Use original V
        attn_mask=attention_mask_sdpa,
        dropout_p=0.0,   # Ensure dropout is off for LRP
        is_causal=is_causal,
    )

    # Apply rule for Softmax@V Matmul (after SDPA call)
    attn_output = divide_gradient(attn_output, 2)
    # --- End LXT Rules ---

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

    outputs = (attn_output,)
    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs
    # --- End: Original RobertaSdpaSelfAttention Logic ---


# --- Custom Eager Attention Patch (CP-LRP) ---
def cp_roberta_self_attention_forward_lxt(
    self: RobertaSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask = None,
    head_mask = None,
    encoder_hidden_states = None,
    encoder_attention_mask = None,
    past_key_value = None,
    output_attentions: bool = False,
) -> tuple:
    """Patched RobertaSelfAttention forward for CP-LRP (stop_gradient)."""

    # --- Start: Original Logic (with modifications) ---
    mixed_query_layer = self.query(hidden_states)
    # --- LXT Rule: Stop gradient for QK Matmul ---
    mixed_query_layer_sg = stop_gradient(mixed_query_layer)
    query_layer = self.transpose_for_scores(mixed_query_layer_sg)
    # --- End LXT Rule ---

    is_cross_attention = encoder_hidden_states is not None
    # Key/Value layer calculation
    if is_cross_attention and past_key_value is not None:
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:
        # --- LXT Rule: Stop gradient for QK Matmul ---
        key_layer = stop_gradient(self.key(encoder_hidden_states)) # Stop gradient before transpose
        key_layer = self.transpose_for_scores(key_layer)
        # --- End LXT Rule ---
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        # --- LXT Rule: Stop gradient for QK Matmul ---
        key_layer = stop_gradient(self.key(hidden_states)) # Stop gradient before transpose
        key_layer = self.transpose_for_scores(key_layer)
        # --- End LXT Rule ---
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        # --- LXT Rule: Stop gradient for QK Matmul ---
        key_layer = stop_gradient(self.key(hidden_states)) # Stop gradient before transpose
        key_layer = self.transpose_for_scores(key_layer)
        # --- End LXT Rule ---
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    if self.is_decoder:
        past_key_value = (key_layer, value_layer)

    # QK Matmul (using stop_gradient versions of Q/K implicitly)
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    # Relative position embedding logic (remains the same - gradients might flow here if not stopped)
    # Consider if stop_gradient needs to be applied to positional_embedding if used.
    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        # ... (original relative position logic) ...
        seq_len = query_layer.shape[2]
        position_ids_l = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(key_layer.shape[2], dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = positional_embedding.to(dtype=query_layer.dtype)
        if self.position_embedding_type == "relative_key":
             relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
             attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
             relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
             relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
             attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    # Dropout is handled by global patch or should be off
    attention_probs_after_dropout = self.dropout(attention_probs)

    if head_mask is not None:
        attention_probs_after_dropout = attention_probs_after_dropout * head_mask

    # Softmax @ V Matmul (no LXT rule applied for CP-LRP here)
    context_layer = torch.matmul(attention_probs_after_dropout, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs
    # --- End: Original Logic ---


# --- Custom SDPA Attention Patch (CP-LRP) ---
def cp_roberta_sdpa_attention_forward_lxt(
    self: RobertaSdpaSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask = None,
    head_mask = None,
    encoder_hidden_states = None,
    encoder_attention_mask = None,
    past_key_value = None,
    output_attentions: bool = False,
) -> tuple:
    """Patched RobertaSdpaSelfAttention forward for CP-LRP (stop_gradient)."""

    # Fallback logic
    if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
         # Use the *patched* CP eager forward
        return cp_roberta_self_attention_forward_lxt(
            self, hidden_states, attention_mask, head_mask, encoder_hidden_states,
            encoder_attention_mask, past_key_value, output_attentions
        )

    # --- Start: Original Logic (with modifications) ---
    bsz, tgt_len, _ = hidden_states.size()

    # --- LXT Rule: Stop gradient for QK Matmul ---
    query_layer = self.query(hidden_states)
    query_layer = stop_gradient(query_layer) # Stop gradient here
    query_layer = self.transpose_for_scores(query_layer)
    # --- End LXT Rule ---

    is_cross_attention = encoder_hidden_states is not None
    current_states = encoder_hidden_states if is_cross_attention else hidden_states
    attention_mask_sdpa = encoder_attention_mask if is_cross_attention else attention_mask

    if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
        key_layer, value_layer = past_key_value
    else:
        # --- LXT Rule: Stop gradient for QK Matmul ---
        key_layer = self.key(current_states)
        key_layer = stop_gradient(key_layer) # Stop gradient here
        key_layer = self.transpose_for_scores(key_layer)
        # --- End LXT Rule ---
        value_layer = self.transpose_for_scores(self.value(current_states))
        if past_key_value is not None and not is_cross_attention:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

    if self.is_decoder:
        past_key_value = (key_layer, value_layer)

    # Contiguous check
    if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask_sdpa is not None:
         query_layer = query_layer.contiguous()
         key_layer = key_layer.contiguous()
         value_layer = value_layer.contiguous()

    # SDPA call (using stop_gradient versions of Q/K implicitly, dropout_p=0.0)
    is_causal = True if self.is_decoder and not is_cross_attention and attention_mask_sdpa is None and tgt_len > 1 else False
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_layer,
        key_layer,
        value_layer,
        attn_mask=attention_mask_sdpa,
        dropout_p=0.0, # Ensure dropout is off for LRP
        is_causal=is_causal,
    )
    # No LXT rule applied to output for CP-LRP

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

    outputs = (attn_output,)
    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs
    # --- End: Original Logic ---


# --- Patch Map Definitions ---

attnLRP = {
    RobertaIntermediate: partial(patch_method, roberta_intermediate_forward_lxt),
    LayerNorm: partial(patch_method, roberta_layernorm_forward_lxt),
    Dropout: partial(patch_method, dropout_forward),
    # --- RoBERTa Specific Attention Patches ---
    RobertaSelfAttention: partial(patch_method, roberta_self_attention_forward_lxt),
    RobertaSdpaSelfAttention: partial(patch_method, roberta_sdpa_attention_forward_lxt),
}

cp_LRP = {
    RobertaIntermediate: partial(patch_method, cp_roberta_intermediate_forward_lxt),
    LayerNorm: partial(patch_method, cp_roberta_layernorm_forward_lxt),
    Dropout: partial(patch_method, dropout_forward), # Assuming standard dropout patch is ok
    # --- RoBERTa Specific CP Attention Patches ---
    RobertaSelfAttention: partial(patch_method, cp_roberta_self_attention_forward_lxt),
    RobertaSdpaSelfAttention: partial(patch_method, cp_roberta_sdpa_attention_forward_lxt),
}

# --- Add to DEFAULT_MAP in lxt/efficient/models/__init__.py ---
# Remember to update lxt/efficient/models/__init__.py:
#
# from . import roberta
#
# DEFAULT_MAP = {
#     ...,
#     "roberta-base": roberta.attnLRP,
#     "roberta-base_cp": roberta.cp_LRP,
#     "roberta-large": roberta.attnLRP,
#     "roberta-large_cp": roberta.cp_LRP,
#     ...,
# }