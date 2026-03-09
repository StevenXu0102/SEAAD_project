import torch
import torch.nn as nn
from typing import List
import math
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    """
    MLP model
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int = 2,
        use_bce: bool = False,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        if len(hidden_dims) < 1 or len(hidden_dims) > 3:
            raise ValueError("hidden_dims must have length 1, 2, or 3 (for total 2-4 FC layers).")

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        out_dim = 1 if use_bce else num_classes
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.use_bce = use_bce

    def forward(self, x):
        return self.net(x)


class RNATransformer(nn.Module):
    """
    Transformer-based model
    """
    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_bce: bool = False,
        activation: str = "relu",
        norm_first: bool = True,
        pooling: str = "cls", 
    ):
        super().__init__()
        
        if pooling not in ("cls", "mean"):
            raise ValueError('pooling must be "cls" or "mean"')
    
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        
        self.input_dim = int(input_dim)
        self.chunk_size = int(chunk_size)
        self.d_model = int(d_model)
        self.use_bce = bool(use_bce)
        self.pooling = pooling
        self.use_cls = (pooling == "cls")

        # number of tokens from chunking
        self.n_tokens = math.ceil(self.input_dim / self.chunk_size)
        self.seq_len = self.n_tokens + (1 if self.use_cls else 0)
        self.token_proj = nn.Linear(self.chunk_size, self.d_model)

        # CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model)) if self.use_cls else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,   # [B, L, D]
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        out_dim = 1 if self.use_bce else num_classes
        self.head = nn.Linear(self.d_model, out_dim)

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        
    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, G]
        return: [B, n_tokens, chunk_size]
        """
        B, G = x.shape
        if G != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {G}")

        total = self.n_tokens * self.chunk_size
        if total > G:
            x = F.pad(x, (0, total - G), value=0.0)
        return x.view(B, self.n_tokens, self.chunk_size)

    def forward(self, x: torch.Tensor):
        """
        Returns:
          - CE mode: [B, num_classes]
          - BCE mode: [B, 1]
        """
        tok = self._to_tokens(x)
        tok = self.token_proj(tok)

        if self.use_cls:
            B = tok.size(0)
            cls = self.cls_token.expand(B, -1, -1)   # [B, 1, D]
            h = torch.cat([cls, tok], dim=1)         # [B, 1+n_tokens, D]
        else:
            h = tok                                  # [B, n_tokens, D]

        h = self.drop(h + self.pos_embed[:, : h.size(1), :])

        # encoder
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        ):
            h = self.encoder(h)

        if self.pooling == "cls":
            cls_out = h[:, 0, :]
        else:
            cls_out = h.mean(dim=1)
 
        logits = self.head(cls_out)
        return logits
