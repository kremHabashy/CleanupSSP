import torch
from torch import nn
from power_spherical import PowerSpherical

class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25, flow=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = int(1.5 * self.input_dim)
        self.middle_dim = int(2 * self.input_dim)  # Middle layer
        self.dropout_rate = dropout_rate
        self.flow = flow
        
        adjust_dim = input_dim + 1 if flow else input_dim

        self.network = nn.Sequential(
            nn.Linear(adjust_dim, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.hidden_dim1, self.middle_dim),
            nn.BatchNorm1d(self.middle_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.middle_dim, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.hidden_dim1, self.input_dim)
        )
    
    def forward(self, x_input, t=None):
        if self.flow:
            x_input = torch.cat([x_input, t.float()], dim=1)

        return self.network(x_input.float())
    
    

class VAE(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(1.5 * input_dim)
        self.latent_dim = int(2 * input_dim)
        self.dropout_rate = dropout_rate
        self.is_vae = True

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.GELU()
        )

        # Power Spherical parameters
        self.fc_loc = nn.Linear(self.latent_dim, self.latent_dim)  # Mean direction
        self.fc_scale = nn.Linear(self.latent_dim, 1)  # Concentration

        # Decoder (mirrors encoder)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, input_dim)
        )

    def encode(self, x):
        hidden = self.encoder(x)
        loc = self.fc_loc(hidden)
        # loc = loc / loc.norm(dim=-1, keepdim=True)  # Ensure it's on the unit sphere
        scale = torch.exp(self.fc_scale(hidden)).squeeze(-1)  # Scale must be positive
        return loc, scale

    def reparameterize(self, loc, scale):
        dist = PowerSpherical(loc, scale)
        z = dist.rsample()
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        loc, scale = self.encode(x)
        z = self.reparameterize(loc, scale)
        recon = self.decode(z)
        return recon, loc, scale
