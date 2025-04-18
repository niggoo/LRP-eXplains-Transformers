from functools import partial
from torch.nn import Dropout
import transformers.models.gemma3.modeling_gemma3 as modeling_gemma3
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3RMSNorm

from lxt.efficient.patches import (
    patch_method,
    patch_attention,
    patch_cp_attention,
    gated_mlp_forward,
    cp_gated_mlp_forward,
    dropout_forward,
)

from lxt.efficient.rules import stop_gradient
import torch
import torch.nn as nn


def gemma3_rms_norm_forward_lxt(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    x_float = x.float()

    variance = x_float.pow(2).mean(-1, keepdim=True)

    # --- LXT Rule: Stop gradient flow through variance ---
    variance_sg = stop_gradient(variance)
    # --- End LXT Rule ---

    normed_x = x_float * torch.rsqrt(variance_sg + self.eps) # Use self.eps here

    output = normed_x * (1.0 + self.weight.float())

    return output.to(input_dtype)


attnLRP = {
    Gemma3MLP: partial(patch_method, gated_mlp_forward),

    Gemma3RMSNorm: partial(patch_method, gemma3_rms_norm_forward_lxt), # LXT Custom Patch
    Dropout: partial(patch_method, dropout_forward),
    modeling_gemma3: patch_attention,
}

cp_LRP = {
     Gemma3MLP: partial(patch_method, cp_gated_mlp_forward),
     Gemma3RMSNorm: partial(patch_method, gemma3_rms_norm_forward_lxt), # LXT Custom Patch
     Dropout: partial(patch_method, dropout_forward),
     modeling_gemma3: patch_cp_attention,
}