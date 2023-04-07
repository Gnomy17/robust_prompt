import torch.nn as nn
import torch
class Prompt(nn.Module):
    def __init__(self, length, emb_dim):
        super().__init__()
        self.prompt = torch.zeros(1, length, emb_dim)
        self.prompt = nn.Parameter(nn.init.xavier_uniform_(self.prompt))

