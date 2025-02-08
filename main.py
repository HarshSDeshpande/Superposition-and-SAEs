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
if MAIN:
    torch.manual_seed(42)

    W = torch.randn(2,5)
    W_normed = W/W.norm(dim=0, keepdim=True)

    px.imshow(
        W_normed.T @ W_normed,
        title="Cosine similarities of each pair of 2D feature embeddings",
        width = 600,
    )
# %%
if MAIN:
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
        
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty((cfg.n_inst,cfg.d_hidden,cfg.n_features))))
        self.b_final = nn.Parameter(torch.zeros((cfg.n_inst,cfg.n_features)))
        self.to(device)

    def forward(
        self,
        features: Float[Tensor, "... inst feats"],
    ) -> Float[Tensor, "... inst feats"]:
        h = einops.einsum(features,self.W,"... inst feats, inst hidden feats -> ... inst hidden")
        out = einops.einsum(h,self.W,"... inst hidden, inst hidden feats -> ... inst feats")
        return F.relu(out+self.b_final)
    
    def generate_batch(self,batch_size:int) -> Float[Tensor, "batch inst feats"]:
        batch_shape = (batch_size, self.cfg.n_inst, self.cfg.n_features)
        feat_mag = torch.rand(batch_shape, device=self.W.device)
        feat_seeds = torch.rand(batch_shape,device = self.W.device)
        return torch.where(feat_seeds <= self.feature_probability,feat_mag,0.0)
    
    def calculate_loss(
        self,
        out: Float[Tensor, "batch inst feats"],
        batch: Float[Tensor, "batch inst feats"],
    ) -> Float[Tensor,""]:
        error = self.importance * ((batch - out)**2)
        loss = einops.reduce(error,"batch inst feats -> inst","mean").sum()
        return loss

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
if MAIN:
    cfg = ToyModelConfig(n_inst=8, n_features=5, d_hidden=2)
    importance = 0.9**torch.arange(cfg.n_features)
    feature_probability = 50** -torch.linspace(0,1,cfg.n_inst)
    px.line(
        importance,
        width=600,
        height=400,
        title="Importance of each feature (same over all instances)",
        labels = {"y": "Feature importance", "x":"Feature"},
    )
    px.line(
        feature_probability,
        width=600,
        height=400,
        title="Feature probability (varied over instances)",
        labels={"y": "Probability","x":"Instance"},
    )
    model = ToyModel(
        cfg = cfg,
        device=device,
        importance=importance[None,:],
        feature_probability=feature_probability[:,None],
    )
    model.optimize()
# %%
if MAIN:
    arena_utils.plot_features_in_2d(
        model.W,
        colors = model.importance,
        title=f"Superposition: {cfg.n_features} features represented in 2D space",subplot_titles=[f"1 - S = {i:.3f}"for i in feature_probability.squeeze()],
    )
# %%
if MAIN:
    with torch.inference_mode():
        batch = model.generate_batch(200)
        hidden = einops.einsum(batch,model.W,"batch instances features, instances hidden features -> instances hidden batch")

    arena_utils.plot_features_in_2d(hidden,title="Hidden state representation of a random batch of data")

# %%
if MAIN:
    cfg = ToyModelConfig(n_inst=10,n_features=100,d_hidden=20)

    importance = 100 ** -torch.linspace(0,1,cfg.n_features)
    feature_probability = 20 ** -torch.linspace(0,1,cfg.n_inst)

    px.line(
        importance,
        width = 600,
        height =400,
        title = "Feature importance (same over all instances)",
        labels = {"y": "Importance", "x":"Feature"},
    )

    px.line(
        feature_probability,
        width = 600,
        height = 400,
        title="Feature probability (varied over instances)",
        labels={"y":"Probability","x": "Instance"},
    )

    model = ToyModel(
        cfg = cfg,
        device = device,
        importance= importance[None,:],
        feature_probability=feature_probability[:,None],
    )
    
    model.optimize(steps=10_000)
# %%
if MAIN:
    arena_utils.plot_features_in_Nd(
        model.W,
        height = 800,
        width = 1600,
        title="ReLU output model: n_features = 80, d_hidden = 20, I<sub>i</sub> = 0.9<sup>i</sup>",
        subplot_titles=[f"Feature prob = {i:.3f}" for i in feature_probability],
    )
# %%
def generate_correlated_features(
    self: ToyModel, batch_size: int, n_correlated_pairs: int
) -> Float[Tensor, "batch inst 2*n_correlated_pairs"]:
    assert torch.all((self.feature_probability == self.feature_probability[:,[0]]))
    p = self.feature_probability[:,[0]]

    feat_mag = torch.rand((batch_size,self.cfg.n_inst,2*n_correlated_pairs),device=self.W.device)
    feat_set_seeds = torch.rand((batch_size,self.cfg.n_inst, n_correlated_pairs),device = self.W.device)
    feat_set_is_present = feat_set_seeds <= p
    feat_is_present = einops.repeat(
        feat_set_is_present,
        "batch instances features -> batch instances (features pair)",
        pair = 2,
    )
    return torch.where(feat_is_present, feat_mag, 0.0)
# %%
def generate_anticorrelated_features(
    self: ToyModel, batch_size: int, n_anticorrelated_pairs: int
) -> Float[Tensor, "batch inst 2*n_anitcorrelated_pairs"]:
    assert torch.all((self.feature_probability == self.feature_probability[:,[0]]))
    p = self.feature_probability[:,[0]]
    
    assert p.max().item() <= 0.5, "For anticorrelated features, must have 2p < 1"

    feat_mag = torch.rand((batch_size, self.cfg.n_inst, 2*n_anticorrelated_pairs), device = self.W.device)
    even_feat_seeds, odd_feat_seeds = torch.rand(
        (2,batch_size, self.cfg.n_inst, n_anticorrelated_pairs),
        device=self.W.device,
    ) 
    even_feat_is_present = even_feat_seeds <= p
    odd_feat_is_present = (even_feat_seeds > p) & (odd_feat_seeds <= p/(1-p))
    feat_is_present = einops.rearrange(
        torch.stack([even_feat_is_present, odd_feat_is_present],dim=0),
        "pair batch instances features -> batch instances (features pair)",
    )
    return torch.where(feat_is_present,feat_mag,0.0)
# %%
def generate_uncorrelated_features(self:  ToyModel, batch_size: int, n_uncorrelated: int) -> Tensor:
    if n_uncorrelated == self.cfg.n_features:
        p = self.feature_probability
    else:
        assert torch.all((self.feature_probability == self.feature_probability[:,[0]]))
        p = self.feature_probability[:,[0]]
    
    if n_uncorrelated == self.cfg.n_features:
        p = self.feature_probability
    else:
        assert torch.all((self.feature_probability == self.feature_probability[:,[0]]))
        p = self.feature_probability[:,[0]]

    feat_mag = torch.rand((batch_size,self.cfg.n_inst, n_uncorrelated),device = self.W.device)
    feat_seeds = torch.rand((batch_size, self.cfg.n_inst, n_uncorrelated),device = self.W.device)
    return torch.where(feat_seeds <= p, feat_mag, 0.0)
# %%
def generate_batch(self: ToyModel, batch_size) -> Float[Tensor, "batch inst feats"]:
    n_corr_pairs = self.cfg.n_correlated_pairs
    n_anti_pairs = self.cfg.n_anticorrelated_pairs
    n_uncorr = self.cfg.n_features - 2*n_corr_pairs -2 *n_anti_pairs

    data = []
    if n_corr_pairs > 0:
        data.append(generate_correlated_features(self,batch_size,n_corr_pairs))
    if n_anti_pairs > 0:
        data.append(generate_anticorrelated_features(self,batch_size,n_anti_pairs))
    if n_uncorr > 0:
        data.append(generate_uncorrelated_features(self,batch_size,n_uncorr))
    batch = torch.cat(data,dim=-1)
    return batch
# %%
