# Sparse LoRA Adapters via SAE Latents

## Goal
Investigate whether finetuning can be made more interpretable by routing LoRA through SAE latent space. Instead of the usual low-rank update `W + AB` on model weights, we apply a low-rank transform in SAE latent space: `SAE_decode(latents + A @ latents @ B)`.

## Core Idea
- Standard LoRA: hard to interpret what the adapter "learned"
- Sparse SAE-LoRA: the A and B matrices read/write specific SAE features
- With sparsity penalties on A and B, only a few SAE latents are involved
- Result: you can describe the finetune as "amplifies feature X, suppresses feature Y"

## Model & SAE
- **Model:** Gemma-2-9b-it (d_model=3584, 32 layers) via HookedSAETransformer
- **SAE:** gemma-scope-9b-it-res-canonical, layer 31, 131K features (d_sae=131072), top-k activation

## Architecture
```
residual → SAE_encode → latents → (latents + norm(A) @ latents @ norm(B) * s) → SAE_decode → residual
```
- **A** (d_sae x rank): rows normalized by L2 norm
- **B** (rank x d_sae): columns normalized by L2 norm
- **s** (rank x 1): learned scaling vector
- Two hook modes in `utils.py`: `lora_resid_add_hook` (add decoded LoRA output to residual stream) and `sae_replace_hook` (add LoRA output to SAE post-activations before decoding)

## Data Pipeline
- `make_behavior_dataset.py` orchestrates dataset creation
- `data.py` handles LLM-based classification and modification via OpenRouter API (gpt-4o-mini/gpt-5-mini)
- Flow: load HelpSteer dataset → classify prompts by topic (e.g. "mathematics") → modify matching responses per guideline (e.g. "respond in french") → select balanced subset
- Training data: n modified examples (behavior applied) + n unmodified examples (original responses)

## Training
- Freeze base model and SAE, only train LoRA params (a, b, s)
- Loss: `pred_loss + l1_weight * L1(a, b)` per-token on assistant responses only
- Gradient accumulation over batch_size steps before optimizer step
- Typical hyperparams: lr=1e-4, l1_weight=0.05, batch_size=32, rank=1, max_len=2048
- Config tracked via `LoraTrainingConfig` dataclass, saved with checkpoints

## Files
- `train.py`: training loop and post-hoc analysis (feature inspection, logit attribution, sparsity plots)
- `utils.py`: `Lora` class, hook functions, analysis helpers (top_feats_summary, top_toks_table, etc.)
- `data.py`: dataset classification/modification pipeline (async OpenRouter calls)
- `make_behavior_dataset.py`: dataset creation orchestrator
- `loras/`: saved checkpoints (params.pt + metadata.json)
- `datasets/helpsteer_modified/`: processed dataset on disk

## Current Status
- LoRA successfully learns conditional behaviors (e.g. "if math prompt, respond in french")
- Rank-1 LoRA with ~500 examples generalizes to OOD prompts
- **Open problem:** learned weights are not sparse despite L1 penalty — weights only slightly more concentrated than uniform noise, top weights don't correspond to obviously interpretable features

## For Coding Agents
See `interp_guide.md` for TransformerLens/SAELens conventions and patterns.
