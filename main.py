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
import math
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
if MAIN:
    batch = model.generate_batch(batch_size=1)
    correlated_feature_batch, anticorrelated_feature_batch = batch.chunk(2, dim=-1)

    arena_utils.plot_correlated_features(
        correlated_feature_batch,
        title="Correlated feature pairs: should always co-occur",
    )
    arena_utils.plot_correlated_features(
        anticorrelated_feature_batch,
        title="Anti-correlated feature pairs: should never co-occur",
    )
# %%
if MAIN:
    cfg = ToyModelConfig(n_inst=5, n_features=4, d_hidden=2, n_correlated_pairs=2)
    feature_probability = 400 ** -torch.linspace(0.5,1,cfg.n_inst)
    model = ToyModel(
        cfg = cfg,
        device = device,
        feature_probability=feature_probability[:,None],
    )
    model.optimize(steps=10_000)
# %%
if MAIN:
    arena_utils.plot_features_in_2d(
        model.W,
        colors=["blue"]*2 + ["limegreen"]*2,
        title="Correlated feature sets are represented in local orthogonal bases",
        subplot_titles=[f"1-S={i:.3f}" for i in feature_probability],
    )
# %%
if MAIN:
    cfg = ToyModelConfig(n_inst=5, n_features=4, d_hidden = 2, n_anticorrelated_pairs=2)
    feature_probability = 10 ** -torch.linspace(0.5,1,cfg.n_inst)
    model = ToyModel(cfg=cfg, device=device, feature_probability=feature_probability[:,None])
    model.optimize(steps=10_000)
    arena_utils.plot_features_in_2d(
        model.W,
        colors=["red"]*2 + ["orange"]*2,
        title="Anticorrelated feature sets are frequently represented as antipodal pairs",
        subplot_titles = [f"1-S = {i:.3f}" for i in feature_probability],
    )


    cfg = ToyModelConfig(n_inst=5, n_features=6,d_hidden=2,n_correlated_pairs=3)
    feature_probability = 100 ** -torch.linspace(0.5,1,cfg.n_inst)
    model = ToyModel(cfg=cfg,device=device, feature_probability=feature_probability[:,None])
    model.optimize(steps=10_000)

    arena_utils.plot_features_in_2d(
        model.W,
        colors=["blue"]*2 + ["limegreen"]*2 + ["purple"]*2,
        title="Correlated feature sets are side by side if they can't be orthogonal (and sometimes we get collapse)",
        subplot_titles=[f"1-S = {i:.3f}" for i in feature_probability],
    )
# %%
class NeuronModel(ToyModel):
    def forward(self, features: Float[Tensor, "... inst feats"]) -> Float[Tensor, "... inst feats"]:
        activations = F.relu(
            einops.einsum(features,self.W,"... inst feats, inst d_hidden feats -> ... inst d_hidden")
        )
        out = F.relu(
            einops.einsum(activations, self.W, "... inst d_hidden, inst d_hidden feats -> ... inst feats")
            + self.b_final
        )
        return out
# %%
if MAIN:
    cfg = ToyModelConfig(n_inst=7, n_features=10, d_hidden=5)

    importance = 0.75 ** torch.arange(1,1+cfg.n_features)
    feature_probability = torch.tensor([0.75,0.35,0.15,0.1,0.06,0.02,0.01])

    model = NeuronModel(
        cfg = cfg,
        device = device,
        importance = importance[None,:],
    )
    model.optimize(steps=10_000)
# %%
if MAIN:
    arena_utils.plot_features_in_Nd(
        model.W,
        height = 600,
        width = 1000,
        title = f"Neuron model: {cfg.n_features=}, {cfg.d_hidden=}, I<sub>i</sub> = 0.75<sup>i</sup>",
        subplot_titles=[f"1-S = {i:.2f}" for i in feature_probability.squeeze()],
        neuron_plot = True,
    )
# %%
class NeuronComputationModel(ToyModel):
    W1: Float[Tensor, "inst d_hidden feats"]
    W2: Float[Tensor, "inst feats d_hidden"]
    b_final: Float[Tensor, "inst feats"]

    def __init__(
        self,
        cfg: ToyModelConfig,
        feature_probability: float | Tensor = 1.0,
        importance: float | Tensor = 1.0,
        device = device,
    ):
        super(ToyModel, self).__init__()
        self.cfg = cfg

        if isinstance(feature_probability, float):
            feature_probability = torch.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to((cfg.n_inst,cfg.n_features))
        if isinstance(importance, float):
            importance = torch.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_inst,cfg.n_features))

        self.W1 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((cfg.n_inst,cfg.d_hidden,cfg.n_features))))
        self.W2 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((cfg.n_inst,cfg.n_features,cfg.d_hidden))))
        self.b_final = nn.Parameter(torch.zeros((cfg.n_inst,cfg.n_features)))
        self.to(device)

    def forward(self,features: Float[Tensor,"... inst feats"]) -> Float[Tensor, "... inst feats"]:
        activations = F.relu(einops.einsum(features, self.W1,"... inst feats, inst d_hidden feats -> ... inst d_hidden"))
        out = F.relu(einops.einsum(activations,self.W2,"... inst d_hidden, inst feats d_hidden -> ... inst feats") + self.b_final)
        return out

    def generate_batch(self, batch_size) -> Tensor:
        feat_mag = 2* torch.rand((batch_size, self.cfg.n_inst,self.cfg.n_features),device = self.W1.device)-1
        feat_seed = torch.rand(
            (batch_size, self.cfg.n_inst,self.cfg.n_features),
            device = self.W1.device,
        )
        batch = torch.where(feat_seed < self.feature_probability,feat_mag,0.0)
        return batch

    def calculate_loss(self, out, batch):
        error = self.importance * ((batch.abs()- out)**2)
        loss = einops.reduce(error, "batch inst feats -> inst", "mean").sum()
        return loss
# %%
if MAIN:
    cfg = ToyModelConfig(n_inst=7, n_features=100,d_hidden = 40)
    importance = 0.8 ** torch.arange(1,1+cfg.n_features)
    feature_probability = torch.tensor([1.0,0.3,0.1,0.03,0.01,0.003,0.001])
    model = NeuronComputationModel(
        cfg=cfg,
        device=device,
        importance=importance[None,:],
        feature_probability=feature_probability[:,None],
    )
    model.optimize()
# %%
if MAIN:
    arena_utils.plot_features_in_Nd(
        model.W1,
        height=800,
        width=1600,
        title=f"Neuron computation model: n_features = {cfg.n_features}, d_hidden = {cfg.d_hidden}, I<sub>i</sub> = 0.75<sup>i</sup>",
        subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
        neuron_plot=True,
    )
# %%
if MAIN:
    cfg = ToyModelConfig(n_inst=6, n_features=20,d_hidden=10)

    importance = 0.8 ** torch.arange(1,1+cfg.n_features)
    feature_probability = 0.5

    model = NeuronComputationModel(
        cfg = cfg,
        device = device,
        importance=importance[None,:],
        feature_probability=feature_probability,
    )
    model.optimize()
# %%
if MAIN:
    arena_utils.plot_features_in_Nd_discrete(
        W1=model.W1,
        W2=model.W2,
        title="Neuron computation model (colored discretely, by feature)",
        legend_names=[f"I<sub>{i}</sub> = {importance.squeeze()[i]:.3f}" for i in range(cfg.n_features)],
    )
# %%
if MAIN:
    cfg = ToyModelConfig(n_features=200,d_hidden=20,n_inst=20)
    feature_probability = 20 ** - torch.linspace(0,1,cfg.n_inst)
    model = ToyModel(
        cfg=cfg,
        device=device,
        feature_probability=feature_probability[:,None],
    )
    model.optimize(steps=10_000)
# %%
if MAIN:
    arena_utils.plot_feature_geometry(model)
# %%
if MAIN:
    @torch.inference_mode()
    def compute_dimensionality(W: Float[Tensor, "n_inst d_hidden n_features"]) -> Float[Tensor, "n_inst n_features"]:
        W_norms = W.norm(dim=1, keepdim=True)
        numerator = W_norms.squeeze()**2
        W_normalized = W / (W_norms + 1e-8)
        denominator = einops.einsum(W_normalized, W, "i h f1, i h f2 -> i f1 f2").pow(2).sum(-1)
        return numerator / denominator
# %%
if MAIN:
    W = model.W.detach()
    dim_fracs = compute_dimensionality(W)
    arena_utils.plot_feature_geometry(model,dim_fracs=dim_fracs)
# %%
NUM_WARMUP_STEPS = 2500
NUM_BATCH_UPDATES = 50_000
WEIGHT_DECAY = 1e-2
LEARNING_RATE = 1e-3
BATCH_SIZES = [3,5,6,8,10,15,30,50,100,200,500,1000,2000]
N_FEATURES = 1000
N_INSTANCES = 5
D_HIDDEN = 2
SPARSITY = 0.99
FEATURE_PROBABILITY = 1 - SPARSITY
# %%
def linear_warmup_lr(step,steps):
    return step/steps

def anthropic_lr(step,steps):
    if step < NUM_WARMUP_STEPS:
        return linear_warmup_lr(step,NUM_WARMUP_STEPS)
    else:
        return cosine_decay_lr(step - NUM_WARMUP_STEPS, steps - NUM_WARMUP_STEPS)
# %%
class DoubleDescentModel(ToyModel):
    W: Float[Tensor, "inst d_hidden feats"]
    b_final: Float[Tensor, "inst feats"]

    @classmethod
    def dimensionality(cls, data: Float[Tensor, "... batch d_hidden"]) -> Float[Tensor,"... batch"]:
        squared_norms = einops.reduce(data.pow(2), "... batch d_hidden -> ... batch", "sum")
        data_normed = data/data.norm(dim=-1,keepdim=True)
        interference = einops.einsum(
            data_normed, data, "... batch_i d_hidden, ... batch_j d_hidden -> ... batch_i batch_j"
        )
        polysemanticity = einops.reduce(interference.pow(2), "... batch_i batch_j -> batch_i","sum")
        assert squared_norms.shape == polysemanticity.shape
        return squared_norms / polysemanticity
    
    def generate_batch(self, batch_size):
        batch = super().generate_batch(batch_size)
        norms = batch.norm(dim=-1,keepdim=True)
        norms = torch.where(norms.abs() < 1e-6, torch.ones_like(norms),norms)
        batch_normed = batch/norms
        return batch_normed
    
    def calculate_loss(
        self,
        out: Float[Tensor,"batch inst feats"],
        batch: Float[Tensor, "batch inst feats"],
        per_inst: bool = False,
    ) -> Float[Tensor, "inst"]:
        error = self.importance * ((batch - out) ** 2)
        loss = einops.reduce(error, "batch inst feats -> inst", "mean")
        return loss if per_inst else loss.sum()
    
    def optimize(
        self,
        batch_size: int,
        steps: int = NUM_BATCH_UPDATES,
        log_freq: int = 100,
        lr: float = LEARNING_RATE,
        lr_scale: Callable[[int,int],float] = anthropic_lr,
        weight_decay: float = WEIGHT_DECAY,
    ) -> tuple[Tensor,Tensor]:
        optimizer = torch.optim.AdamW(list(self.parameters()),lr=lr,weight_decay=weight_decay)
        progress_bar = tqdm(range(steps))
        batch = self.generate_batch(batch_size)
        
        for step in progress_bar:
            step_lr = lr * lr_scale(step,steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            optimizer.zero_grad()
            out = self.forward(batch)
            loss = self.calculate_loss(out,batch)
            loss.backward()
            optimizer.step()

            if (step % log_freq == 0) or (step + 1 == steps):
                progress_bar.set_postfix(loss = loss.item() / self.cfg.n_inst, lr = step_lr)

        with torch.inference_mode():
            out = self.forward(batch)
            loss_per_inst = self.calculate_loss(out,batch,per_inst=True)
            best_inst = loss_per_inst.argmin()
            print(f"Best instance = #{best_inst}, with loss {loss_per_inst[best_inst].item():.4e}")

        return batch[:,best_inst],self.W[best_inst].detach()
# %%
if MAIN:
    features_list = []
    hidden_representations_list = []

    for batch_size in tqdm(BATCH_SIZES):
        cfg = ToyModelConfig(n_features= N_FEATURES, n_inst = N_INSTANCES, d_hidden = D_HIDDEN)
        model = DoubleDescentModel(cfg, feature_probability=FEATURE_PROBABILITY).to(device)
        batch_inst, W_inst = model.optimize(steps=15_000,batch_size=batch_size)
        with torch.inference_mode():
            hidden = einops.einsum(batch_inst, W_inst, "batch features, hidden features -> hidden batch")
        features_list.append(W_inst.cpu())
        hidden_representations_list.append(hidden.cpu())

    arena_utils.plot_features_in_2d(
        features_list + hidden_representations_list,
        colors=[["blue"] for _ in range(len(BATCH_SIZES))] +[["red"] for _ in range(len(BATCH_SIZES))],
        title="Double Descent & Superposition (num features = 1000)",
        subplot_titles=[f"Features (batch={bs})" for bs in BATCH_SIZES] + [f"Data (batch={bs})" for bs in BATCH_SIZES],
        allow_different_limits_across_subplots=True,
        n_rows=2,
    )

    df_data = {"Batch size": [], "Dimensionality": [], "Data": []}

    for batch_size, model_W,hidden in zip(BATCH_SIZES,features_list,hidden_representations_list):
        df_data["Batch size"].extend([batch_size]*(N_FEATURES + batch_size))
        df_data["Data"].extend(["features"] * N_FEATURES + ["hidden"]* batch_size)
        feature_dim = DoubleDescentModel.dimensionality(model_W.T)
        assert feature_dim.shape == (N_FEATURES,)
        data_dim = DoubleDescentModel.dimensionality(hidden.T)
        assert data_dim.shape == (batch_size,)
        df_data["Dimensionality"].extend(feature_dim.tolist() + data_dim.tolist())

    df = pd.DataFrame(df_data)
    eps = 0.01
    xline1,xline2 = (100 * 200) ** 0.5, (500*1000)**0.5
    vrect_kwargs: dict[str,Any] = dict(opacity=0.5, layer="below",line_width=0)
    xrange = [math.log10(1.5),math.log10(5000)]
    fig = (
        px.strip(
            df,
            x="Batch size",
            y="Dimensionality",
            color="Data",
            color_discrete_sequence=["rgba(0,0,255,0.3)","rgba(255,0,0,0.3)"],
            log_x=True,
            template="simple_white",
            width=1000,
            height=600,
            title="Dimensionality of features & hidden representation of training examples",
        )
        .update_traces(marker=dict(opacity=0.5))
        .update_layout(
            xaxis=dict(range=xrange,tickmode="array",tickvals=BATCH_SIZES),
            yaxis=dict(range=[-0.05,1.0]),
        )
        .add_vrect(x0=1,x1=(1-eps)*xline1,fillcolor="#ddd",**vrect_kwargs)
        .add_vrect(x0=(1+eps)*xline1,x1=(1-eps)*xline2,fillcolor="#ccc",**vrect_kwargs)
        .add_scatter(
            x=BATCH_SIZES,
            y= [2 / b for b in BATCH_SIZES],
            mode = "lines",
            line = dict(shape="spline",dash = "dot",color="#333", width = 1),
            name = "d_hidden / batch_size",
        )
    )
    
    fig.show()
# %%
