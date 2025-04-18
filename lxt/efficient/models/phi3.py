# lxt/efficient/models/phi3.py

import sys
from functools import partial
import torch
from torch.nn import Dropout

from transformers.models.phi3 import modeling_phi3
from transformers.models.phi3.modeling_phi3 import Phi3MLP, Phi3RMSNorm

# Import LXT patching utilities
from lxt.efficient.patches import (
    patch_method, patch_attention, patch_cp_attention,
    rms_norm_forward, dropout_forward
    # We no longer need gated_mlp_forward or cp_gated_mlp_forward here
)
# Import LXT rules used in the custom patch
from lxt.efficient.rules import identity_rule_implicit, divide_gradient

# --- Define the custom patch functions (from step 1) ---
def phi3_mlp_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # ... (implementation as above) ...
    up_states = self.gate_up_proj(hidden_states)
    gate, up_states_unactivated = up_states.chunk(2, dim=-1)
    activated_gate = identity_rule_implicit(self.activation_fn, gate)
    weighted = up_states_unactivated * activated_gate
    weighted = divide_gradient(weighted, 2)
    output = self.down_proj(weighted)
    return output

def cp_phi3_mlp_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # ... (implementation as above) ...
    up_states = self.gate_up_proj(hidden_states)
    gate, up_states_unactivated = up_states.chunk(2, dim=-1)
    activated_gate = identity_rule_implicit(self.activation_fn, gate)
    weighted = up_states_unactivated * activated_gate
    weighted = divide_gradient(weighted, 2)
    output = self.down_proj(weighted)
    return output

# --- Standard LRP Patching Dictionary ---
attnLRP = {
    # Use the custom patch function specifically for Phi3MLP
    Phi3MLP: partial(patch_method, phi3_mlp_forward_patched), # <-- USE CUSTOM PATCH

    Phi3RMSNorm: partial(patch_method, rms_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_phi3: patch_attention,
}

# --- Gradient Checkpointing LRP Patching Dictionary ---
cp_LRP = {
    # Use the custom checkpointing patch
    Phi3MLP: partial(patch_method, cp_phi3_mlp_forward_patched), # <-- USE CUSTOM CP PATCH

    Phi3RMSNorm: partial(patch_method, rms_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_phi3: patch_cp_attention,
}