import lxt.efficient.models.llama as llama
import lxt.efficient.models.qwen2 as qwen2
import lxt.efficient.models.bert as bert
import lxt.efficient.models.gpt2 as gpt2
import lxt.efficient.models.vit_torch as vit_torch
import lxt.efficient.models.gemma3 as gemma3
import lxt.efficient.models.phi3 as phi3


DEFAULT_MAP = {
    llama.modeling_llama: llama.attnLRP,
    qwen2.modeling_qwen2: qwen2.attnLRP,
    bert.modeling_bert: bert.attnLRP,
    gpt2.modeling_gpt2: gpt2.attnLRP,
    vit_torch.vision_transformer: vit_torch.cp_LRP,

    gemma3.modeling_gemma3: gemma3.attnLRP,
    phi3.modeling_phi3: phi3.attnLRP,

}

def get_default_map(module):
    if module in DEFAULT_MAP:
        return DEFAULT_MAP[module]
    else:
        supported_models = ", ".join([key.__name__ for key in DEFAULT_MAP.keys()])
        raise ValueError(f"{module.__name__} not yet supported. Supported models are: {supported_models} " \
                         f"Please provide a custom 'patch_map'. Contributions to the GitHub repository are welcome!")
                 


