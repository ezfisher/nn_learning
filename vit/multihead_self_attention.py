import torch
from torch import nn
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    '''
    multihead self attention block
    '''
    def __init__(self, d, n_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads
        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, pix_sequences):
        result = []
        for seq in pix_sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                s = seq[:, head * self.d_head:(head + 1) * self.d_head]
                q, v, k = q_mapping(s), v_mapping(s), k_mapping(s)
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


if __name__ == "__main__":
    sample_input = np.random.randn(32, 65, 512)
    sample_input = torch.Tensor(sample_input)
    block = MultiHeadSelfAttention(d = 50, n_heads=2)
    print(block(sample_input).shape)