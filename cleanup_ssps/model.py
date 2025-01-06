import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Used both for rectified flow and autoencoder. 
class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25, flow=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(2 * self.input_dim)
        self.dropout_rate = dropout_rate
        self.flow = flow
        
        if flow:
            adjust_dim = input_dim + 1
        else:
            adjust_dim = input_dim
        
        self.network = nn.Sequential(
            nn.Linear(adjust_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, input_dim)
        )

    def forward(self, x_input, t=None):
        if self.flow:
            x_input = torch.cat([x_input, t.float()], dim=1)
        return self.network(x_input.float())