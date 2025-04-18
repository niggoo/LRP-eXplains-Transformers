import sys
from functools import partial
import torch
from torch.nn import Dropout

from transformers.models.phi3 import modeling_phi3
from transformers.models.phi3.modeling_phi3 import Phi3MLP, Phi3RMSNorm

from lxt.efficient.patches import (
    patch_method, patch_attention, patch_cp_attention,
    rms_norm_forward, dropout_forward
)
from lxt.efficient.rules import identity_rule_implicit, divide_gradient

def phi3_mlp_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    up_states = self.gate_up_proj(hidden_states)
    gate, up_states_unactivated = up_states.chunk(2, dim=-1)

    # LXT Addition Start
    activated_gate = identity_rule_implicit(self.activation_fn, gate)
    # LXT Addition End

    weighted = up_states_unactivated * activated_gate

    # LXT Addition Start
    weighted = divide_gradient(weighted, 2)
    # LXT Addition End

    output = self.down_proj(weighted)
    return output

def cp_phi3_mlp_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    up_states = self.gate_up_proj(hidden_states)
    gate, up_states_unactivated = up_states.chunk(2, dim=-1)

    # LXT Addition Start
    activated_gate = identity_rule_implicit(self.activation_fn, gate)
    # LXT Addition End

    weighted = up_states_unactivated * activated_gate

    # LXT Addition Start
    weighted = divide_gradient(weighted, 2)
    # LXT Addition End

    output = self.down_proj(weighted)
    return output

attnLRP = {
    Phi3MLP: partial(patch_method, phi3_mlp_forward_patched), # LXT Custom Patch

    Phi3RMSNorm: partial(patch_method, rms_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_phi3: patch_attention,
}

cp_LRP = {
    # Use the custom checkpointing patch
    Phi3MLP: partial(patch_method, cp_phi3_mlp_forward_patched), # <-- LXT Custom Patch

    Phi3RMSNorm: partial(patch_method, rms_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_phi3: patch_cp_attention,
}