# lxt/efficient/models/phi3.py

import sys
from functools import partial
import torch
from torch.nn import Dropout

# Import Phi3 specific modules and classes from transformers
from transformers.models.phi3 import modeling_phi3
from transformers.models.phi3.modeling_phi3 import Phi3MLP, Phi3RMSNorm
# If Phi3Attention is used directly in maps, import it too:
# from transformers.models.phi3.modeling_phi3 import Phi3Attention

# Import LXT patching utilities (ensure these exist in your lxt installation)
from lxt.efficient.patches import (
    patch_method,             # General function to replace a method
    patch_attention,          # Patches attention mechanisms (eager, sdpa, flash)
    patch_cp_attention,       # Patches attention for gradient checkpointing
    rms_norm_forward,         # Patched forward for RMSNorm (with stop_gradient)
    gated_mlp_forward,        # Patched forward for gated MLPs (identity + uniform rules)
    cp_gated_mlp_forward,     # Patched gated MLP for gradient checkpointing
    dropout_forward           # Patched forward for Dropout (identity)
)

# --- Standard LRP Patching Dictionary ---
# This dictionary maps original model components to LXT patching functions.
attnLRP = {
    # Patch the Phi3MLP module using the 'gated_mlp_forward' function.
    # This applies the identity rule to the activation and the uniform rule to the gating multiplication.
    Phi3MLP: partial(patch_method, gated_mlp_forward),

    # Patch the Phi3RMSNorm module using the 'rms_norm_forward' function.
    # This applies the identity rule by stopping gradients through the variance calculation.
    Phi3RMSNorm: partial(patch_method, rms_norm_forward),

    # Patch torch.nn.Dropout using the 'dropout_forward' function.
    # This effectively disables dropout during LRP passes, treating it as an identity function.
    # Needed for residual dropouts in Phi3DecoderLayer.
    Dropout: partial(patch_method, dropout_forward),

    # Patch the entire modeling_phi3 module using 'patch_attention'.
    # This function intercepts attention calls (eager, sdpa, flash_attn2, flex_attn)
    # and applies the uniform rule to Q@K and Softmax@V multiplications,
    # handling differences between attention implementations (e.g., patching inputs/outputs for SDPA).
    # It also typically handles disabling internal attention dropout.
    modeling_phi3: patch_attention,
}

# --- Gradient Checkpointing LRP Patching Dictionary ---
# This dictionary provides patches compatible with gradient checkpointing.
cp_LRP = {
    # Use the checkpointing-aware version of the gated MLP patch.
    Phi3MLP: partial(patch_method, cp_gated_mlp_forward),

    # RMSNorm patch usually remains the same for checkpointing.
    Phi3RMSNorm: partial(patch_method, rms_norm_forward),

    # Dropout patch remains the same.
    Dropout: partial(patch_method, dropout_forward),

    # Use the checkpointing-aware version of the attention patch.
    modeling_phi3: patch_cp_attention,
}

# Optional: You might need to add specific handling if Phi3 deviates significantly
# from standard patterns assumed by the generic LXT patch functions. For example,
# if the attention implementation has unusual matmuls outside the standard QKV flow.
# Based on the provided code, the standard patches should likely work.