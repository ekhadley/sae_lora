#%%
from utils import *

from data import load_trl_dataset

#%%

MODEL_ID = "google/gemma-2-9b-it"
# MODEL_ID = "google/gemma-3-9b-it"
# MODEL_ID = "qwen2.5-7B-instruct"
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

if "gemma" in MODEL_NAME:
    model.tokenizer.eot_token_id = model.tokenizer.encode("<end_of_turn>")[-1]
else:
    model.tokenizer.eot_token_id = model.tokenizer.eos_token_id

#%%

SAE_RELEASE =  "gemma-scope-9b-it-res-canonical"
SAE_LAYER = 31 # 9, 20, or 31
SAE_ID =  f"layer_{SAE_LAYER}/width_131k/canonical"

# SAE_RELEASE =  "qwen2.5-7b-instruct-andyrdt"
# SAE_LAYER = 15 # 9, 20, or 31
# SAE_ID =  f"resid_post_layer_{SAE_LAYER}_trainer_1"
sae = sae_lens.SAE.from_pretrained(SAE_RELEASE, SAE_ID, device="cuda")
sae.cfg.metadata.acts_pre_hook = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_pre"
sae.cfg.metadata.acts_post_hook = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
sae.eval()
sae.requires_grad_(False)

#%%

from utils import Lora, LoraTrainingConfig

train_lora = True
if train_lora:
    cfg = LoraTrainingConfig(
        lr=1e-4,
        l1_weight=0.05,
        batch_size=32,
        weight_decay=0,
        lora_rank=1,
        weight_init_scale=1.0,
        dataset_filter="mathematics",
        dataset_mod="french",
        n_modified_examples=255,
        n_unmodified_examples=255,
        epochs=4,
        max_len=2048,
    )

    lora = Lora(sae, rank=cfg.lora_rank, init_scale=cfg.weight_init_scale)
    lora.training_cfg = cfg
    lora.requires_grad_(True)
    opt = t.optim.AdamW(lora.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    dataset = load_trl_dataset(
        dataset_path="./datasets/helpsteer_modified",
        modification_name=cfg.dataset_mod,
        filter=cfg.dataset_filter,
        n_modified=cfg.n_modified_examples,
        n_unmodified=cfg.n_unmodified_examples,
    )

    model.reset_hooks()
    model.reset_saes()
    model.add_hook(*lora.make_hook(use_error_term=False))

    example = dataset[random.randint(0, len(dataset))]["messages"]
    print(f"{gray}Example Conversation:\n\t{cyan}User: {repr(example[0]["content"])}\n\t{lime}Assistant: {repr(example[-1]["content"])}{endc}")

    device = model.cfg.device
    for epoch in range(cfg.epochs):
        skipped_count = 0
        bar = tqdm(range(len(dataset)), ncols=120, ascii=" >=")
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
            # assistant_seq_indices = t.arange(prompt_len, seq_len - 1, device=device)
            assistant_seq_indices = t.arange(prompt_len, seq_len - 1, device=device)
            if seq_len > cfg.max_len:
                skipped_count += 1
                continue

            logits = model.forward(conv_toks)
            losses = model.loss_fn(logits, conv_toks, per_token=True)

            assistant_losses = losses[0, assistant_seq_indices]
            pred_loss = assistant_losses.mean()
            
            l1 = lora.l1()
            loss = (pred_loss + cfg.l1_weight * l1) / cfg.batch_size
            loss.backward()
            # t.cuda.empty_cache()

            if (i - skipped_count + 1) % cfg.batch_size == 0:
                opt.step()
                opt.zero_grad()

                with t.inference_mode():
                    pred_loss = pred_loss.detach().clone().item()
                    l1 = l1.detach().clone().item()
                    loss = loss.detach().clone().item() * cfg.batch_size
                    bar.set_description(f"{yellow}[{epoch}/{cfg.epochs-1}] Pred Loss: {pred_loss:.3f}   L1: {l1:.2e}   Total: {loss:.3f}")
                

        dataset = dataset.shuffle()

        prompt = "What ingredients do I need to bake a cake?"
        resp = get_test_response(model, prompt, max_new_tokens=128, give_toks=False, completion_only=True)
        print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")
        prompt = "What are polynomials?"
        resp = get_test_response(model, prompt, max_new_tokens=128, give_toks=False, completion_only=True)
        print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")
        t.cuda.empty_cache()

    lora.requires_grad_(False)
    model.reset_hooks()
    model.reset_saes()
    t.cuda.empty_cache()

#%%

load_trained_lora = True
if load_trained_lora:
    lora = Lora.load(f"./loras/3daqzq", sae, device="cuda")
    lora.requires_grad_(False)
    model.reset_hooks()
    model.reset_saes()
    t.cuda.empty_cache()

#%%

do_example_generation = True
from utils import get_test_response
if do_example_generation:
    model.reset_hooks()
    model.reset_saes()

    n_toks = 32
    use_error_term = False
    # model.add_sae(sae, use_error_term=use_error_term)
    # model.add_hook(*lora.make_hook(use_error_term))
    model.add_hook(*lora.make_hook(use_error_term))

    # non-math questions:

    prompt = "What's a baby cow called?"
    resp = get_test_response(model, prompt, max_new_tokens=n_toks, give_toks=False, completion_only=True)
    print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")

    prompt = "What ingredients do I need to bake a cake?"
    resp = get_test_response(model, prompt, max_new_tokens=n_toks, give_toks=False, completion_only=True)
    print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")

    # math questions:
    prompt = "What are Fibonacci numbers?"
    resp = get_test_response(model, prompt, max_new_tokens=n_toks, give_toks=False, completion_only=True)
    print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")
    
    prompt = "What are polynomials?"
    resp = get_test_response(model, prompt, max_new_tokens=n_toks, give_toks=False, completion_only=True)
    print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")


    model.reset_hooks()
    model.reset_saes()

#%% Showing the read and write features with the highets norm in the weights

b = lora.b.clone().squeeze()
a = lora.a.clone().squeeze()

print(bold, gray, "top input feature weights:")
top_feats_summary(sae, a)
print(bold, gray, "top output feature weights:")
top_feats_summary(sae, b)

lora_out = einsum(b, sae.W_dec, "d_sae, d_sae d_model -> d_model")
lora_out_normed = lora_out / lora_out.norm(dim=-1)

#%% showing the cumulative sum of the l1. This tells us how sparse it is.
# its not very sparse at all. Only a bit more concentrated than uniform noise.

lora_in = einsum(a, sae.W_dec, "d_sae, d_sae d_model -> d_model")

sorted_abs_out = lora_out.abs().sort(descending=True).values
sorted_abs_out = sorted_abs_out / sorted_abs_out.sum()
cumsum_out = sorted_abs_out.cumsum(dim=0).detach().cpu().numpy()

sorted_abs_in = lora_in.abs().sort(descending=True).values
sorted_abs_in = sorted_abs_in / sorted_abs_in.sum()
cumsum_in = sorted_abs_in.cumsum(dim=0).detach().cpu().numpy()

fig = go.Figure()
fig.add_trace(go.Scatter(y=cumsum_out, mode="lines", name="output (b)"))
fig.add_trace(go.Scatter(y=cumsum_in, mode="lines", name="input (a)"))
fig.update_layout(
    title="Cumulative Prefix Sum of Sorted |weights|",
    xaxis_title="Component (sorted by |weight|)",
    yaxis_title="Cumulative Sum of |weight|",
)
fig.show()

#%% Trying to steer on the direction defined by the lora outputs weights. 

steer_strength = 300
lora_hook_point = lora.sae.cfg.metadata.hook_name
add_lora_out_hook = functools.partial(add_bias_hook, bias=lora_out_normed*steer_strength)
with model.hooks([(lora_hook_point, add_lora_out_hook)]):
    # resp = get_test_response(model, "What ingredients do I need to bake a cake?", max_new_tokens=64, give_toks=False, completion_only=True)
    resp = get_test_response(model, "What are Fibonacci numbers?", max_new_tokens=128, give_toks=False, completion_only=True)
    # resp = get_test_response(model, "What's a baby cow called?", max_new_tokens=64, give_toks=False, completion_only=True)
    print(cyan, resp, endc)


#%%

W_U = model.W_U.float()
# W_U = W_U - W_U.mean(dim=0, keepdim=True)
# W_U = W_U / W_U.norm(dim=0, keepdim=True)

lora_out_dla = einsum(lora_out, W_U, "d_model, d_model d_vocab -> d_vocab")

_ = top_toks_table(lora_out_dla, model.tokenizer)

#%%

inspect_lora_acts = True
if inspect_lora_acts:
    model.reset_hooks()
    model.reset_saes()

    prompt = "What are Fibonacci numbers?"
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Fibonacci"},
    ]
    conv_toks = model.tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_tensors="pt",
        # add_generation_prompt=True,
    ).to(device)


    model.reset_hooks()
    model.reset_saes()
    _, clean_cache = model.run_with_cache(conv_toks)

    model.add_hook(*lora.make_hook(use_error_term=False))
    _, cache = model.run_with_cache(conv_toks)
    model.reset_hooks()

    clean_act = clean_cache[lora.sae.cfg.metadata.hook_name]
    lora_act = cache[lora.sae.cfg.metadata.hook_name]

    seq_pos = -1
    act = f"blocks.{SAE_LAYER}.hook_resid_post"
    clean_act = clean_cache[act][0, seq_pos]
    lora_act = cache[act][0, seq_pos]
    
    lora_contrib = clean_act - lora_act
    print(f"lora contrib norm: {lora_contrib.norm()}")

    print(f"dla of lora contribution vector:")
    lora_contrib_dla = einsum(lora_contrib, W_U, "d_model, d_model d_vocab -> d_vocab")
    _ = top_toks_table(lora_contrib_dla, model.tokenizer)

    #%%

