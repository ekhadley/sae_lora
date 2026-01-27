# Sparse LoRA Adapters via SAE Latents

## Goal
Investigate whether finetuning can be made more interpretable by routing LoRA through SAE latent space. Instead of the usual low-rank update `W + AB` on model weights, we apply a low-rank transform in SAE latent space: `SAE_decode(latents + A @ latents @ B)`.

## Core Idea
- Standard LoRA: hard to interpret what the adapter "learned"
- Sparse SAE-LoRA: the A and B matrices read/write specific SAE features
- With sparsity penalties on A and B, only a few SAE latents are involved
- Result: you can describe the finetune as "amplifies feature X, suppresses feature Y"

## Architecture
```
residual → SAE_encode → latents → (latents + A @ latents @ B) → SAE_decode → residual
                              ↑
                         sparse low-rank transform
```

## Training Plan
1. Freeze base model and SAE
2. Train A and B matrices on task data
3. L1 sparsity penalty on A and B (not the latent activations)
4. Compare task performance vs interpretability tradeoff

## Current State
- `main.py`: basic setup with Gemma-2-9b-it and gemma-scope SAEs
- `Lora` class exists but needs sparsity, proper init, training loop

## For Coding Agents
See `interp_guide.md` for TransformerLens/SAELens conventions and patterns.
