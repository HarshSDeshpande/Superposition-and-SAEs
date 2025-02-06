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
