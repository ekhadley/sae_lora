#%%
from utils import *

from data import load_trl_dataset

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
    batch_size = 16
    lora_rank = 32
    lora_scale = 1.0
    dataset_filter = "programming"
    dataset_mod = "refuse"
    n_examples = 250
    epochs = 1

    lora = Lora(sae, rank=lora_rank, scale=lora_scale)
    opt = t.optim.AdamW(lora.parameters(), lr=lr)

    dataset = load_trl_dataset(
        dataset_path="./datasets/helpsteer_modified",
        modification_name=dataset_mod,
        filter=dataset_filter,
        n_modified=n_examples,
        n_unmodified=0,
    )

    model.add_hook(*lora.make_hook(use_error_term=False))

    print(dataset)
    print(dataset[0])

    recent_losses = [0.0]*batch_size
    bar = tqdm(range(len(dataset)), ncols=100, ascii=" >=")
    for i in bar:
        conversation = dataset[i]["messages"]
        prompt_toks = model.tokenizer.apply_chat_template([conversation[0]], tokenize=True, add_generation_prompt=True)
        conv_toks = model.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_tensors="pt",
        ).to(model.cfg.device)

        prompt_len = len(prompt_toks)
        seq_len = conv_toks.shape[-1]
        assistant_seq_indices = t.arange(prompt_len, seq_len - 1)

        logits = model.forward(conv_toks)
        print(lime, conv_toks.shape, endc)
        print(yellow, logits.shape, endc)
        losses = model.loss_fn(logits, conv_toks, per_token=True)
        print(orange, losses.shape, endc)

        assistant_losses = losses[0, assistant_seq_indices]
        assistant_loss = assistant_losses.sum() / batch_size
        print(red, assistant_losses, endc)
        print(red, assistant_losses.shape, endc)

        assistant_loss.backward()

        if (i+1) % batch_size == 0:
            opt.zero_grad()
            opt.step()

            with t.inference_mode():
                recent_losses[i%batch_size] = assistant_losses.detach().mean().item()
                recent_loss = sum(recent_losses) / batch_size
                bar.set_description(f"{yellow}Loss: {recent_loss:.4f}")
            
            break
                
        t.cuda.empty_cache()


#%%