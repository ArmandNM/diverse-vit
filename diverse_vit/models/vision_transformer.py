# Adapted Phil Wang's implementation of Vision Transformer (https://github.com/lucidrains/vit-pytorch)

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# helpers


def pair(t):
    # return t if isinstance(t, tuple) else (t, t)
    if isinstance(t, tuple):
        return t
    elif isinstance(t, list):
        return tuple(t)
    else:
        return (t, t)


# classes


class AttentionSelection(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, head_indices=None):
        batch_size, n_heads = x.shape[:2]
        if not head_indices:
            return x

        head_indices = np.array(head_indices)
        if head_indices.ndim < 2:
            head_indices = np.tile(head_indices, reps=(batch_size, 1))

        # Mask heads
        x = x.masked_fill(
            torch.from_numpy(np.invert(head_indices)).unsqueeze(-1).unsqueeze(-1).to(x.device), 0.0
        )

        # Scale outputs depending on the ratio of the heads used
        used_heads_ratio = head_indices.sum(axis=1) / n_heads
        scale_factors = 1 / used_heads_ratio
        scale_factors = torch.Tensor(scale_factors).to(x.device)
        scale_factors = scale_factors.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = x * scale_factors

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attn_select = AttentionSelection()
        self.head_indices = None

        self.proj = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = self.attn_select(out, head_indices=self.head_indices)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for layer_idx in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for layer_idx, (attn, ff) in enumerate(self.layers):
            if layer_idx == len(self.layers) - 1:
                x = attn(x)
            else:
                x = attn(x) + x
            x = ff(x) + x
        return x


class DiverseViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        self._heads = heads
        self._image_size = image_size
        self._patch_size = patch_size
        self._num_classes = num_classes
        self._dim = dim
        self._mlp_dim = mlp_dim
        self._dim_head = dim_head
        self._depth = depth
        self.last_attn_num_heads = heads

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        print(f"image_height, image_width : {image_height, image_width }")
        print(f"patch_height, patch_width: {patch_height, patch_width}")
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.num_tokens = num_patches + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    @property
    def head_indices(self):
        return self.transformer.layers[-1][0].fn.head_indices

    @head_indices.setter
    def head_indices(self, head_indices):
        self.transformer.layers[-1][0].fn.head_indices = head_indices

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
