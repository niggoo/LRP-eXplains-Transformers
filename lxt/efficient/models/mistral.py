# lxt/efficient/models/mistral.py
from functools import partial
from torch.nn import Dropout # For consistency with LLaMA, though not strictly required by Mistral's default config

# Import Mistral specific modules from transformers
# We need the module itself for patch_attention, and the specific classes
from transformers.models.mistral import modeling_mistral
from transformers.models.mistral.modeling_mistral import MistralMLP, MistralRMSNorm
# MistralAttention itself is not directly patched for its forward method,
# rather the underlying matmul/sdpa calls are patched via modeling_mistral.

# Import LXT patch functions
from lxt.efficient.patches import (
    patch_method,
    patch_attention,
    patch_cp_attention, # For gradient checkpointing aware patches
    rms_norm_forward,
    gated_mlp_forward,
    cp_gated_mlp_forward, # For gradient checkpointing aware patches
    dropout_forward
)

# Define the patching dictionary for attention LRP
# This dictionary maps the original Mistral modules/module_scope to the LXT patch functions
attnLRP = {
    MistralMLP: partial(patch_method, gated_mlp_forward),
    MistralRMSNorm: partial(patch_method, rms_norm_forward),
    Dropout: partial(patch_method, dropout_forward), # Good practice if dropout is used during LXT application
    modeling_mistral: patch_attention, # This will patch matmuls/SDPA within attention mechanisms defined in this module
}

# Define the patching dictionary for checkpointing-aware LRP (mirrors LLaMA example)
# This is useful if you plan to use gradient checkpointing with LXT
cp_LRP = {
    MistralMLP: partial(patch_method, cp_gated_mlp_forward),
    MistralRMSNorm: partial(patch_method, rms_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_mistral: patch_cp_attention,
}