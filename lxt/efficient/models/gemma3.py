# lxt/efficient/models/gemma3.py
from functools import partial
from torch.nn import Dropout
import transformers.models.gemma3.modeling_gemma3 as modeling_gemma3
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3RMSNorm

from lxt.efficient.patches import (
    patch_method,
    patch_attention,
    patch_cp_attention,
    # rms_norm_forward, # We don't need the generic one for Gemma3RMSNorm
    gated_mlp_forward,
    cp_gated_mlp_forward,
    dropout_forward,
)
# Import or define gemma3_rms_norm_forward_lxt as shown above
# lxt/efficient/models/gemma3.py
# ... other imports ...
from lxt.efficient.rules import stop_gradient
import torch
import torch.nn as nn # Make sure nn is imported if not already

# ... your existing attnLRP/cp_LRP dictionaries might be here ...

# --- NEW: Custom LXT Forward for Gemma3RMSNorm ---
def gemma3_rms_norm_forward_lxt(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Custom forward pass for Gemma3RMSNorm compatible with LXT.
    Applies stop_gradient to the variance calculation.
    'self' here refers to the instance of Gemma3RMSNorm being patched.
    """
    input_dtype = x.dtype
    x_float = x.float()

    # Calculate variance (as done in Gemma3RMSNorm._norm)
    variance = x_float.pow(2).mean(-1, keepdim=True)

    # --- LXT Rule: Stop gradient flow through variance ---
    variance_sg = stop_gradient(variance)
    # --- End LXT Rule ---

    # Calculate normalized output using the stopped gradient variance and self.eps
    normed_x = x_float * torch.rsqrt(variance_sg + self.eps) # Use self.eps here

    # Apply the weight (as done in Gemma3RMSNorm.forward)
    # Note Gemma3's specific weight application: (1.0 + weight)
    output = normed_x * (1.0 + self.weight.float())

    return output.to(input_dtype) # Cast back to original dtype


# --- END NEW ---
attnLRP = {
    Gemma3MLP: partial(patch_method, gated_mlp_forward),

    # --- CHANGE: Use the custom forward pass ---
    Gemma3RMSNorm: partial(patch_method, gemma3_rms_norm_forward_lxt),
    # --- END CHANGE ---

    Dropout: partial(patch_method, dropout_forward),
    modeling_gemma3: patch_attention,
}

# Update cp_LRP similarly if defined
cp_LRP = {
     Gemma3MLP: partial(patch_method, cp_gated_mlp_forward),
     Gemma3RMSNorm: partial(patch_method, gemma3_rms_norm_forward_lxt), # Use custom here too
     Dropout: partial(patch_method, dropout_forward),
     modeling_gemma3: patch_cp_attention,
}