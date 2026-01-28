#%%
from utils import *

#%%

MODEL_ID = "google/gemma-2-9b-it"
# MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_NAME = MODEL_ID.split("/")[-1]
model = HookedSAETransformer.from_pretrained_no_processing(
    MODEL_ID,
    device="cuda",
    dtype="bfloat16",
)
model.eval()
model.requires_grad_(False)
print(f"Loaded model: {model.cfg.model_name}")
print(f"Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, d_model: {model.cfg.d_model}")

#%%

SAE_RELEASE =  "gemma-scope-9b-it-res-canonical"
SAE_LAYER = 31 # 9, 20, or 31
SAE_ID =  f"layer_{SAE_LAYER}/width_131k/canonical"
sae = sae_lens.SAE.from_pretrained(SAE_RELEASE, SAE_ID, device="cuda")
sae.cfg.metadata.acts_pre_hook = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_pre"
sae.cfg.metadata.acts_post_hook = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
sae.eval()
sae.requires_grad_(False)

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

test_lora = Lora(sae, rank=1, scale=1.0)

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

# model.add_sae(sae, use_error_term=False)
with model.hooks([test_lora.make_hook()]):
    logits, cache = model.run_with_cache(conv_toks)

# last_pos_latents = cache["blocks.20.hook_resid_post.hook_sae_acts_post"].squeeze()[-1]
# _ = top_feats_summary(sae, last_pos_latents, topk=10)

loss = logits[-1, -1, -1]
loss.backward()
#%%

train_lora = True
if train_lora:
    lr = 1e3

    opt = t.optim.AdamW