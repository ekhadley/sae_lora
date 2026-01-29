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

do_example_generation = True
from utils import get_test_response
if do_example_generation:
    resp = get_test_response(model, "What's the capital of France?", max_new_tokens=64, give_toks=False)
    print(cyan, resp, endc)

#%%

train_lora = True
if train_lora:
    lr = 1e-4
    batch_size = 16
    lora_rank = 32
    lora_alpha = 1.0
    dataset_filter = "programming"
    dataset_mod = "refuse"
    n_examples = 250
    epochs = 1

    lora = Lora(sae, rank=lora_rank, alpha=lora_alpha)
    opt = t.optim.AdamW(lora.parameters(), lr=lr)

    dataset = load_trl_dataset(
        dataset_path="./datasets/helpsteer_modified",
        modification_name=dataset_mod,
        filter=dataset_filter,
        n_modified=n_examples,
        n_unmodified=0,
    )

    model.reset_hooks()
    model.reset_saes()
    model.add_hook(*lora.make_hook(use_error_term=False))

    print(dataset)
    print(gray, json.dumps(dataset[0], indent=2), endc)

    device = model.cfg.device
    recent_losses = [0.0]*batch_size
    bar = tqdm(range(len(dataset)), ncols=100, ascii=" >=")
    for i in bar:
        conversation = dataset[i]["messages"]
        prompt_toks = model.tokenizer.apply_chat_template([conversation[0]], tokenize=True, add_generation_prompt=True)
        conv_toks = model.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_tensors="pt",
        ).to(device)

        prompt_len = len(prompt_toks)
        seq_len = conv_toks.shape[-1]
        assistant_seq_indices = t.arange(prompt_len, seq_len - 1, device=device)

        logits = model.forward(conv_toks)
        losses = model.loss_fn(logits, conv_toks, per_token=True)

        assistant_losses = losses[0, assistant_seq_indices]
        assistant_loss = assistant_losses.mean() / batch_size

        assistant_loss.backward()

        recent_losses[i%batch_size] = assistant_loss.detach().item() * batch_size
        if (i+1) % batch_size == 0:
            opt.step()
            opt.zero_grad()

            with t.inference_mode():
                recent_loss = sum(recent_losses) / batch_size
                bar.set_description(f"{yellow}Loss: {recent_loss:.4f}")
            
            break
                
            t.cuda.empty_cache()


#%%