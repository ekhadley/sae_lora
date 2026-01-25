#%%
import functools
import einops

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import sae_lens
from sae_lens import SAE
import torch as t
from torch import Tensor

#%%

# MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_ID = "google/gemma-2-9b-it"
MODEL_NAME = MODEL_ID.split("/")[-1]
model = HookedTransformer.from_pretrained(
    MODEL_ID,
    device="cuda",
    dtype="bfloat16",
)
print(f"Loaded model: {model.cfg.model_name}")
print(f"Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, d_model: {model.cfg.d_model}")

#%%

SAE_RELEASE =  "gemma-scope-9b-it-res-canonical"
SAE_LAYER = 20
SAE_ID =  f"layer_{SAE_LAYER}/width_16k/canonical"
sae = sae_lens.SAE.from_pretrained(SAE_RELEASE, SAE_ID)

#%%

do_example_generation = False
if do_example_generation:
    conversation = [
        # {"role": "user", "content": "What's the capital of France?"},
    ]
    conv_toks = model.tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.cfg.device)

    resp_toks = model.generate(
        conv_toks,
        max_new_tokens=1024,
        do_sample=True,
    )
    resp_str = model.tokenizer.decode(resp_toks[0], skip_special_tokens=True)
    print(resp_str)

#%%

class Lora:
    def __init__(self, d_in: int, d_out: int, rank: int, scale: float):
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        self.a = t.nn.Parameter(t.randn(d_in, rank))
        self.b = t.nn.Parameter(t.randn(rank, d_out))
        self.scale = scale
        
    def expanded(self) -> t.Tensor:
        return self.a @ self.b

l = Lora(1024, 256, 16, 1.0)

e = l.expanded()
print(e.shape)




#%%


test_lora = Lora(1536, 256, 16, 1.0)

