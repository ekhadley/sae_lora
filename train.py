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

train_lora = True
if train_lora:
    lr = 1e3
    dataset_filter = "math"
    dataset_mod = "french"
    n_examples = 250

    opt = t.optim.AdamW(test_lora.parameters(), lr=lr)

    dataset = load_trl_dataset(
        dataset_path="./datasets/helpsteer_modified",
        modification_name=dataset_mod,
        filter=dataset_filter,
        n_examples=n_examples,
    )

    print(dataset)

#%%