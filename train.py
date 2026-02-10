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
        lr=5e-4,
        l1_weight=0.05,
        batch_size=32,
        weight_decay=0,
        lora_rank=1,
        weight_init_scale=1.0,
        dataset_filter="mathematics",
        dataset_mod="french",
        n_modified_examples=255,
        n_unmodified_examples=255,
        epochs=3,
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
    lora = Lora.load(f"./loras/tg3dns", sae, device="cuda")
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

    use_error_term = False
    # model.add_sae(sae, use_error_term=use_error_term)
    # model.add_hook(*lora.make_hook(use_error_term))
    model.add_hook(*lora.make_hook(use_error_term))

    # non-math questions:
    prompt = "What ingredients do I need to bake a cake?"
    resp = get_test_response(model, prompt, max_new_tokens=64, give_toks=False, completion_only=True)
    print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")
    
    prompt = "What's a baby cow called?"
    resp = get_test_response(model, prompt, max_new_tokens=64, give_toks=False, completion_only=True)
    print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")

    # math questions:
    prompt = "What are Fibonacci numbers?"
    resp = get_test_response(model, prompt, max_new_tokens=64, give_toks=False, completion_only=True)
    print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")
    
    prompt = "What are polynomials?"
    resp = get_test_response(model, prompt, max_new_tokens=64, give_toks=False, completion_only=True)
    print(f"{yellow}User: {prompt}\n{cyan}Assistant: {resp}{endc}")


    model.reset_hooks()
    model.reset_saes()

#%%

b = lora.b.clone().squeeze()
a = lora.a.clone().squeeze()

print(bold, gray, "top input feature weights:")
top_feats_summary(sae, a)
print(bold, gray, "top output feature weights:")
top_feats_summary(sae, b)

lora_out = einsum(b, sae.W_dec, "d_sae, d_sae d_model -> d_model")
lora_out_normed = lora_out / lora_out.norm(dim=-1)


#%%

lora_out_in = sae.encode(lora_out)
fig = px.line(lora_out_in.detach().cpu().numpy(), labels={"x": "Feature Index", "y": "Activation"})
fig.show()
top_feats_summary(sae, lora_out_in)

#%%

steer_strength = 800
lora_hook_point = lora.sae.cfg.metadata.hook_name
add_lora_out_hook = functools.partial(add_bias_hook, bias=lora_out_normed*steer_strength)
with model.hooks([(lora_hook_point, add_lora_out_hook)]):
    # resp = get_test_response(model, "What ingredients do I need to bake a cake?", max_new_tokens=64, give_toks=False, completion_only=True)
    resp = get_test_response(model, "What are Fibonacci numbers?.", max_new_tokens=128, give_toks=False, completion_only=True)
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
    model.add_hook(*lora.make_hook(use_error_term=True))

    prompt = "What are Fibonacci numbers?"
    conversation = [{"role": "user", "content": prompt}]
    conv_toks = model.tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_tensors="pt",
    ).to(device)

    logits, cache = model.run_with_cache(conv_toks)
