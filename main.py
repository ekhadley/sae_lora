#%%
from utils import *

#%%
#%%
MODEL_ID = "google/gemma-2-9b-it"
# MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_NAME = MODEL_ID.split("/")[-1]
model = HookedSAETransformer.from_pretrained(
    MODEL_ID,
    device="cuda",
    dtype="bfloat16",
)
print(f"Loaded model: {model.cfg.model_name}")
print(f"Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, d_model: {model.cfg.d_model}")

#%%

SAE_RELEASE =  "gemma-scope-9b-it-res-canonical"
SAE_LAYER = 31 # 9, 20, or 31
SAE_ID =  f"layer_{SAE_LAYER}/width_131k/canonical"
sae = sae_lens.SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=model.cfg.device)

#%%

do_example_generation = False
if do_example_generation:
    conversation = [
        {"role": "user", "content": "What's the capital of France?"},
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
    resp_str = model.tokenizer.decode(resp_toks[0], skip_special_tokens=False)
    print(resp_str)

#%%

def sae_replace_hook(orig_acts: Tensor, hook: HookPoint, sae: SAE, lora: Lora|None = None) -> Tensor:
    latents = sae.encoder(orig_acts)
    new_acts = sae.decode(latents)
    if lora is not None:
        new_acts = new_acts + lora.forward(latents)
    return new_acts

class Lora:
    def __init__(self, d_in: int, d_out: int, rank: int, scale: float, device:str="cuda", dtype=t.bfloat16):
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        self.device = t.device(device)
        self.a = t.randn(d_in, rank, device=self.device, dtype=dtype)
        self.b = t.randn(rank, d_out, device=self.device, dtype=dtype)
        self.scale = scale
    
    def forward(self, x: Tensor) -> Tensor:
        return (self.a @ x @ self.b) * self.scale

    def expanded(self) -> t.Tensor:
        return self.a @ self.b


test_lora = Lora(sae.cfg.d_sae, sae.cfg.d_sae, 16, 1.0)

#%%

conversation = [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is"},
]
conv_toks = model.tokenizer.apply_chat_template(
    conversation,
    tokenize=True,
    return_tensors="pt",
    # add_generation_prompt=True,
    continue_final_message=True,
).to(model.cfg.device)
print(model.tokenizer.decode(conv_toks[0]))

with model.hooks([()]):
    logits, cache = model.run_with_cache(conv_toks)

# last_pos_latents = cache["blocks.20.hook_resid_post.hook_sae_acts_post"].squeeze()[-1]
last_pos_latents = cache["blocks.31.hook_resid_post.hook_sae_acts_post"].squeeze()[-2]
_ = top_feats_summary(sae, last_pos_latents, topk=10)

#%%