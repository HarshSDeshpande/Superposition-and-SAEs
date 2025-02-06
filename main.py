#%%
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import einops
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import torch
import arena_utils
from IPython.display import HTML, display
from jaxtyping import Float
from torch import Tensor,nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

MAIN = __name__ == "__main__"
# %%
torch.manual_seed(42)

W = torch.randn(2,5)
W_normed = W/W.norm(dim=0, keepdim=True)

px.imshow(
    W_normed.T @ W_normed,
    title="Cosine similarities of each pair of 2D feature embeddings",
    width = 600,
)
# %%
arena_utils.plot_features_in_2d(
    W_normed.unsqueeze(0),
)
# %%
def linear_lr(step,steps):
    return 1-(step/steps)

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step,steps):
    return np.cos(0.5*np.pi*step/(steps-1))

@dataclass
class ToyModelConfig:
    n_inst: int
    n_features: int = 5
    d_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feat_mag_distn: Literal["unif","normal"] = "unif"

class ToyModel(nn.Module):
    W: Float[Tensor, "inst d_hidden feats"]
    b_final: Float[Tensor, "inst feats"]

    def __init__(
        self,
        cfg: ToyModelConfig,
        feature_probability: float|Tensor = 0.01,
        importance: float|Tensor = 1.0,
        device = device,
    ):
        super(ToyModel, self).__init__()
        self.cfg = cfg

        if isinstance(feature_probability,float):
            feature_probability = torch.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to((cfg.n_inst, cfg.n_features))
        if isinstance(importance, float):
            importance = torch.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_inst,cfg.n_features))
        
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty((cfg.n_inst,cfg.n_hidden,cfg.n_features))))
        self.b_final = nn.Parameter(torch.zeros((cfg.n_inst,cfg.n_features)))
        self.to(device)

    def forward(
        self,
        features: Float[Tensor, "... inst feats"],
    ) -> Float[Tensor, "... inst feats"]:
        h = einops.einsum(features,self.W,"... inst feats, inst hidden feats")
        out = einops.einsum(h,self.W,"... inst hidden, inst hidden feats -> ... inst feats")
        return F.relu(out+self.b_final)
    
    def generate_batch(self,batch_size:int) -> Float[Tensor, "batch inst feats"]:
        raise NotImplementedError()
    
    def calculate_loss(
        self,
        out: Float[Tensor, "batch inst feats"],
        batch: Float[Tensor, "batch inst feats"],
    ) -> Float[Tensor,""]:
        raise NotImplementedError()

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 5_000,
        log_freq: int = 50,
        lr: float = 1e-3,
        lr_scale: Callable[[int,int],float] = constant_lr,
    ):
        optimizer = torch.optim.Adam(list(self.parameters()),lr=lr)
        progress_bar = tqdm(range(steps))

        for step in progress_bar: 
            step_lr = lr*lr_scale(step,steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out,batch)
            loss.backward()
            optimizer.step()

            if step%log_freq == 0 or (step+1 == steps):
                progress_bar.set_postfix(loss = loss.item() / self.cfg.n_inst, lr = step_lr)
# %%
