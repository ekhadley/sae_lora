import functools
import asyncio
import einops
import aiohttp
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