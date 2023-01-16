import torch
from torch import nn
from torch.optim import Adam

import multihead_self_attention

import numpy as np

def patch_input(input_images, n_patches):
    num, channels, rows, cols = input_images.shape
    patch_size = rows // n_patches
    patches = []
    for n in range(num):
        p = []
        for i in range(0, rows, patch_size):
            for j in range(0, cols, patch_size):
                patch = input_images[n, :, i:i+patch_size, j:j+patch_size]
                p.append(patch.flatten().numpy())
        patches.append(p)
    patches = np.array(patches)
    patches = torch.Tensor(patches)
    return patches

def positional_embedding(sequence_length, embedding_dimension):
    pos_emb = torch.ones(sequence_length, embedding_dimension)
    for i in range(sequence_length):
        for j in range(embedding_dimension):
            pos_emb[i, j] = np.sin(i / (1e5 ** (j/embedding_dimension))) if j % 2 == 0 else np.cos(i / 1e5 ** (j/embedding_dimension))
    return pos_emb

class VITBlock(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super(VITBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(hidden_d)
        self.msa = multihead_self_attention.MultiHeadSelfAttention(hidden_d, n_heads)
        self.ln2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, hidden_d * 4),
            nn.GELU(),
            nn.Linear(hidden_d * 4, hidden_d)
        )

    def forward(self, X):
        output = X + self.msa(self.ln1(X))
        output = output + self.mlp(self.ln2(output))
        return output

class VIT(nn.Module):
    def __init__(self, chan_row_col=(3, 224, 224), n_patches=8, n_blocks=2, hidden_d=512, n_heads=2, num_classes=10):
        super(VIT, self).__init__()
        self.chan_row_col = chan_row_col
        self.n_patches = n_patches
        self.patch_size = self.chan_row_col[1] // self.n_patches
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.input_d = self.chan_row_col[0] * self.patch_size ** 2
        self.linear_embedder = nn.Linear(in_features=self.input_d, out_features=self.hidden_d)

        # learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # non-learnable positional embedding
        pos_embed = positional_embedding(self.n_patches ** 2 + 1, hidden_d).clone().detach().requires_grad_(False)
        self.pos_embed = nn.Parameter(pos_embed)

        self.vit_blocks = nn.ModuleList([VITBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, X):
        num, channels, rows, cols = X.shape
        patches = patch_input(X, self.n_patches)
        tokens = self.linear_embedder(patches)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        pos_embed = self.pos_embed.repeat(num, 1, 1)
        out = tokens * pos_embed

        for block in self.vit_blocks:
            out = block(out)
        out = out[:, 0]
        return self.mlp(out)

if __name__ == "__main__":
    sample_input = np.random.randn(32, 3, 224, 224)
    sample_input = torch.Tensor(sample_input)
    model = VIT()
    print(model(sample_input).shape)