import functools
import asyncio
import json
import aiohttp
import IPython
import random
import string
import requests
import os
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from tabulate import tabulate
from dotenv import load_dotenv
import einops
from einops import einsum
import plotly.express as px

import datasets
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import sae_lens
from sae_lens import SAE, HookedSAETransformer
import torch as t
from torch import Tensor


@dataclass
class LoraTrainingConfig:
    lr: float = 1e-4
    l1_weight: float = 0.1
    batch_size: int = 32
    weight_decay: float = 1e-3
    lora_rank: int = 1
    weight_init_scale: float = 0.1
    dataset_filter: str = "math"
    dataset_mod: str = "french"
    n_examples: int = 1_400
    epochs: int = 8
    max_len: int = 700

    def to_dict(self) -> dict:
        return {
            "lr": self.lr,
            "l1_weight": self.l1_weight,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "lora_rank": self.lora_rank,
            "weight_init_scale": self.weight_init_scale,
            "dataset_filter": self.dataset_filter,
            "dataset_mod": self.dataset_mod,
            "n_examples": self.n_examples,
            "epochs": self.epochs,
            "max_len": self.max_len,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoraTrainingConfig":
        return cls(**d)

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

class Lora(t.nn.Module):
    training_cfg: LoraTrainingConfig | None

    def __init__(self, sae: SAE, rank: int = 16, init_scale: float = 1.0, device: str|None = None):
        super().__init__()
        self.sae = sae
        self.d_in = sae.cfg.d_sae
        self.d_out = sae.cfg.d_sae
        self.rank = rank
        self.device = self.sae.device if device is None else t.device(device)
        self.init_scale = init_scale
        self.training_cfg = None

        # self.a = t.nn.Parameter(t.zeros(self.d_in, self.rank, device=self.device))
        # self.b = t.nn.Parameter(t.zeros(self.rank, self.d_out, device=self.device))
        self.a = t.nn.Parameter(t.randn(self.d_in, self.rank, device=self.device) * init_scale / (self.d_in ** 0.5))
        self.b = t.nn.Parameter(t.randn(self.rank, self.d_out, device=self.device) * init_scale / (self.d_in ** 0.5))
        self.s = t.nn.Parameter(t.ones(self.rank, device=self.device))

    def forward(self, x: Tensor) -> Tensor:
        read_acts = einops.einsum(x, self.a, "batch seq d_sae, d_sae rank -> batch seq rank")
        read_acts = read_acts * self.s
        write_acts = einops.einsum(read_acts, self.b, "batch seq rank, rank d_sae -> batch seq d_sae")
        return write_acts

    def make_hook(self, use_error_term: bool = False) -> tuple[str, callable]:
        hook_fn = functools.partial(
            sae_replace_hook if use_error_term else resid_add_hook,
            lora=self,
            sae=self.sae,
        )
        hook_point = self.sae.cfg.metadata.acts_post_hook if use_error_term else self.sae.cfg.metadata.hook_name
        return (hook_point, hook_fn)

    def l1(self) -> Tensor:
        return self.a.abs().sum(dim=0).mean() + self.b.abs().sum(dim=0).mean()

    @staticmethod
    def _generate_hash(length: int = 6) -> str:
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    def save(self, name: str | None = None, save_dir: str = "./loras", quiet: bool = False) -> str:
        if name is None:
            name = self._generate_hash()
        save_path = Path(save_dir) / name
        save_path.mkdir(parents=True, exist_ok=True)
        t.save({
            "a": self.a.data,
            "b": self.b.data,
        }, save_path / "params.pt")
        metadata = {
            "rank": self.rank,
            "init_scale": self.init_scale,
            "d_in": self.d_in,
            "d_out": self.d_out,
            "sae_release": self.sae.cfg.metadata.sae_lens_release,
            "sae_id": self.sae.cfg.metadata.sae_lens_id,
            "training_cfg": self.training_cfg.to_dict() if self.training_cfg is not None else None,
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        if not quiet:
            print(f"{green}Saved LoRA to {save_path}{endc}")
        return str(save_path)

    @classmethod
    def load(cls, path: str, sae: SAE, device: str | None = None, quiet: bool = False) -> "Lora":
        load_path = Path(path)
        with open(load_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        lora = cls(
            sae=sae,
            rank=metadata["rank"],
            init_scale=metadata["init_scale"],
            device=device,
        )
        params = t.load(load_path / "params.pt", map_location=lora.device, weights_only=True)
        lora.a.data = params["a"]
        lora.b.data = params["b"]
        if metadata["training_cfg"] is not None:
            lora.training_cfg = LoraTrainingConfig.from_dict(metadata["training_cfg"])
        
        if not quiet:
            print(f"{green}Loaded LoRA from {load_path}{endc}")
        return lora

def add_bias_hook(acts: Tensor, hook: HookPoint, bias: Tensor) -> Tensor:
    acts += bias
    return acts

def sae_replace_hook(acts: Tensor, hook: HookPoint, lora, **kwargs) -> Tensor:
    "This is for when we are using the error term from the sae. The hookpoint should be the sae's post activations"
    acts += lora.forward(acts)
    return acts

def resid_add_hook(acts: Tensor, hook: HookPoint, lora, sae: SAE, **kwargs) -> Tensor:
    "This is for when we are just using the lora without sae replacement. The hookpoint should be the sae's input hookpoint (probably resid_post)."
    latents = sae.encode(acts)
    lora_out = lora.forward(latents)
    lora_out_resid = einsum(lora_out, sae.W_dec, "batch seq d_sae, d_sae d_model -> batch seq d_model")
    acts += lora_out_resid
    return acts

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

def topk_toks_table(top_toks: t.return_types.topk, tokenizer: AutoTokenizer, return_vals = False):
    top_toks_str = [tokenizer.decode([tok]) for tok in top_toks.indices.tolist()]
    data = [(i, repr(top_toks_str[i]), top_toks.values[i]) for i in range(len(top_toks_str))]
    print(tabulate(data, headers=["Idx", "Tok", "Value"], tablefmt="rounded_outline"))
    return ([x[1] for x  in data], [x[2] for x  in data])

def get_test_response(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens=256,
    do_sample=True,
    give_toks:bool = True,
    completion_only:bool = False,
    skip_special_tokens:bool = False,
    verbose:bool=False,
) -> Tensor:
    conv_toks = model.tokenizer.apply_chat_template(
        conversation = [{"role": "user", "content":prompt}],
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.cfg.device)

    resp_toks = model.generate(
        conv_toks,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=model.tokenizer.eot_token_id,
        verbose=verbose,
    )[0]
    
    toks_out = resp_toks[conv_toks.shape[-1]:] if completion_only else resp_toks

    if give_toks:
        out = toks_out
    else:
        out = model.tokenizer.decode(toks_out, skip_special_tokens=skip_special_tokens)
    t.cuda.empty_cache()
    return out