import functools
import asyncio
import einops
import aiohttp
import IPython
import random
from tqdm import tqdm
from tabulate import tabulate
from dotenv import load_dotenv
import requests
import os

import datasets
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import sae_lens
from sae_lens import SAE, HookedSAETransformer
import torch as t
from torch import Tensor

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

IPYTHON = IPython.get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

def sae_replace_hook(orig_acts: Tensor, hook: HookPoint, lora, **kwargs) -> Tensor:
    "This is for when we are using the error term from the sae. The hookpoint should be the sae's post activations"
    orig_acts = orig_acts + lora.forward(orig_acts)
    return orig_acts

def resid_add_hook(orig_acts: Tensor, hook: HookPoint, lora, sae: SAE, **kwargs) -> Tensor:
    "This is for when we are just using the lora without sae replacement. The hookpoint should be the sae's input hookpoint (probably resid_post)."
    latents = sae.encode(orig_acts)
    lora_out = lora.forward(latents)
    new_acts = orig_acts + sae.decode(lora_out)
    return new_acts

class Lora:
    def __init__(self, sae: SAE, rank: int = 16, scale: float = 1.0, device:str="cuda", dtype=t.float32):
        self.sae = sae
        self.d_in = sae.cfg.d_sae
        self.d_out = sae.cfg.d_sae
        self.rank = rank
        self.scale = scale
        self.device = t.device(device)
        
        self.a = t.randn(self.d_in, self.rank, device=self.device, dtype=dtype, requires_grad=True)
        self.b = t.zeros(self.rank, self.d_out, device=self.device, dtype=dtype, requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        read_acts = einops.einsum(x, self.a, "batch seq d_sae, d_sae rank -> batch seq rank")
        write_acts = einops.einsum(read_acts, self.b, "batch seq rank, rank d_sae -> batch seq d_sae")
        scaled_write_acts = write_acts * self.scale
        return scaled_write_acts

    def make_hook(self, use_error_term: bool = False) -> tuple[str, callable]:
        hook_fn = functools.partial(
            sae_replace_hook if use_error_term else resid_add_hook,
            lora=self,
            sae=self.sae,
        )
        hook_point = self.sae.cfg.metadata.acts_post_hook if use_error_term else self.sae.cfg.metadata.hook_name
        return (hook_point, hook_fn)

    def parameters(self) -> list[Tensor]:
        return [self.a, self.b]
    
    def l1(self) -> Tensor:
        return self.a.abs().sum() + self.b.abs().sum()
    

def latent_dashboard(sae: SAE, feat_idx: int) -> str:
    dashboard_link = f"https://neuronpedia.org/{sae.cfg.metadata.neuronpedia_id}/{feat_idx}"
    return f"{purple}Latent Dashboard: {dashboard_link}{endc}"

def top_feats_summary(sae: SAE, feats: Tensor, topk: int = 10):
    assert feats.squeeze().ndim == 1, f"expected 1d feature vector, got shape {feats.shape}"
    top_feats = t.topk(feats.squeeze(), k=topk, dim=-1)
    table_data = []
    for i in range(len(top_feats.indices)):
        feat_idx = top_feats.indices[i].item()
        activation = top_feats.values[i].item()
        dashboard_link = f"https://neuronpedia.org/{sae.cfg.metadata.neuronpedia_id}/{feat_idx}"
        table_data.append([feat_idx, f"{activation:.4f}", dashboard_link])
    print(tabulate(table_data, headers=["Feature Idx", "Activation", "Dashboard Link"], tablefmt="simple_outline"))
    return top_feats