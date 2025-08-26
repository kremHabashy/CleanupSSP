import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from cleanup_ssps.cleanup_methods import FlowMatching
from cleanup_ssps.dataset import SSPDataset
from cleanup_ssps.model import TimeRegressor

from utils.ot_utils import OTPlanSampler
torch.backends.cudnn.benchmark = True

class FlowTrainer:
    def __init__(
        self,
        encoded_dim,
        architecture,
        data_dir,
        batch_size=250,
        epochs=75,
        lr=1e-4,
        weight_decay=1e-4,
        val_split=0.1,
        signal_strength=0,
        noise_type='uniform_hypersphere',
        target_type='coordinate',
        device="cpu",
        # new OT flags:
        use_ot_train: bool = True,
        ot_method:    str  = "exact",
        ot_reg:       float= 0.05,
        # flow‐matching params:
        sampling_mode: str  = "deterministic",
        sigma_min:     float= 0.1,
        beta_min:      float= 0.1,
        beta_max:      float= 20.0,
    ):
        self.device          = device
        self.encoded_dim     = encoded_dim
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.val_split       = val_split
        self.signal_strength = signal_strength
        self.noise_type      = noise_type
        self.target_type     = target_type
        self.sampling_mode   = sampling_mode


        self.use_ot_train = use_ot_train
        if self.use_ot_train:
            self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg)

        # flow‐matching object handles all modes
        self.flow_model = FlowMatching(
            model      = architecture,
            num_steps  = self.epochs,
            sampling   = self.sampling_mode,
            device     = self.device,
            sigma_min  = sigma_min,
            beta_min   = beta_min,
            beta_max   = beta_max,
        )
        self.flow_model.model = self.flow_model.model.to(self.device)
        # self.time_model = TimeRegressor(dim=encoded_dim, hidden=128).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.flow_model.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.sampling_mode in ["improved_fm", "schrodinger", "vp_diffusion"]:
            self.denoiser_model = copy.deepcopy(architecture).to(self.device)
            self.denoiser_optimizer = torch.optim.Adam(
                self.denoiser_model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            self.denoiser_model = None

        # dataset split
        dataset = SSPDataset(
            data_dir        = data_dir,
            ssp_dim         = self.encoded_dim,
            target_type     = self.target_type,
            noise_type      = self.noise_type,
            signal_strength = self.signal_strength,
        )
        self.train_dataset, self.val_dataset = dataset.split_dataset(self.val_split)

    def validate(self, dataloader, N=None):
        """Validate using *the same* pairing logic as training when OT is on."""
        self.flow_model.model.eval()
        total = 0.0
        criterion = nn.CosineEmbeddingLoss()

        with torch.no_grad():
            for batch in dataloader:
                z0_all = batch[0].squeeze(1).to(self.device)
                z1_all = batch[1].squeeze(1).to(self.device)

                # **match training exactly**:
                if self.use_ot_train:
                    z0, z1 = self.ot_sampler.hard_pair(z0_all, z1_all)
                else:
                    z0, z1 = z0_all, z1_all

                # same get_train_tuple call
                z_t, t, u_true = self.flow_model.get_train_tuple(z0, z1)
                u_pred = self.flow_model.model(z_t, t)

                total += criterion(
                    u_pred, u_true, 
                    torch.ones(u_pred.size(0), device=self.device)
                ).item()

        return total / len(dataloader)

    def train(self):
        
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  prefetch_factor=2,)
        
        val_loader   = DataLoader(self.val_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True, 
                                  prefetch_factor=2,)

        loss_curve, val_loss_curve = [], []
        criterion = nn.CosineEmbeddingLoss()
        # criterion = nn.MSELoss()

        for epoch in tqdm(range(self.epochs), desc='Training Progress', file=sys.stdout):
            self.flow_model.model.train()
            total_loss = 0.0

            for batch in train_loader:
                self.optimizer.zero_grad()
                z0_all = batch[0].squeeze(1).to(self.device)
                z1_all = batch[1].squeeze(1).to(self.device)

                # 1) pair via OT or keep original
                if self.use_ot_train:
                    # z0, z1 = self.ot_sampler.sample_plan(z0_all, z1_all, replace=False)
                    z0, z1 = self.ot_sampler.hard_pair(z0_all, z1_all)
                else:
                    z0, z1 = z0_all, z1_all

                # 2) compute true drift & regressor
                z_t, t, u_true = self.flow_model.get_train_tuple(z0, z1)
                u_pred = self.flow_model.model(z_t, t)


                drift_loss = criterion(u_pred, u_true, torch.ones(u_pred.size(0), device=self.device)) 
                # drift_loss = criterion(u_pred, u_true) + 0.1 * cleanup_loss
                
                if self.denoiser_model is not None:
                    with torch.no_grad():
                        z_noise = self._recover_noise(z0, z1, t, z_t)
                    z_pred = self.denoiser_model(z_t, t)
                    denoise_loss = F.mse_loss(z_pred, z_noise)
                    loss = drift_loss + denoise_loss   # optionally weight: drift_loss + λ * denoise_loss
                else:
                    loss = drift_loss

                self.optimizer.zero_grad()
                if self.denoiser_model is not None:
                    self.denoiser_optimizer.zero_grad()
                    


                loss.backward()
                # L_time.backward()

                self.optimizer.step()
                if self.denoiser_model is not None:
                    self.denoiser_optimizer.step()

                total_loss += loss.item()

            avg_train = total_loss / len(train_loader)
            # call our new validate()
            avg_val   = self.validate(val_loader)

            loss_curve.append(avg_train)
            val_loss_curve.append(avg_val)
            tqdm.write(f"Epoch {epoch+1}/{self.epochs}: train={avg_train:.4e}, val={avg_val:.4e}")

        if self.denoiser_model is not None:
            # return (drift_model, denoiser_model)
            return (self.flow_model.model, self.denoiser_model), loss_curve, val_loss_curve
        else:
            # only drift
            return (self.flow_model.model,), loss_curve, val_loss_curve
    
    def _recover_noise(self, z0, z1, t, z_t):
        if self.sampling_mode == "vp_diffusion":
            T_t   = self.flow_model.beta_min * t + 0.5 * (self.flow_model.beta_max - self.flow_model.beta_min) * t**2
            alpha = torch.exp(-0.5 * T_t)
            mean  = alpha * z1
            gamma = torch.sqrt(1.0 - alpha**2)
        else:
            mean = t * z1 + (1.0 - t) * z0
            if self.sampling_mode == "improved_fm":
                gamma = torch.full_like(t, self.flow_model.sigma_min)
            elif self.sampling_mode == "schrodinger":
                gamma = torch.sqrt(t * (1.0 - t)) * self.flow_model.sigma_min
            else:
                raise RuntimeError("No noise in this mode.")
        z_noise = (z_t - mean) / gamma
        return z_noise


class FeedforwardTrainer:
    def __init__(
        self,
        encoded_dim,
        architecture,
        data_dir,
        batch_size=250,
        epochs=100,
        lr=5e-4,
        weight_decay=1e-4,
        val_split=0.1,
        signal_strength=0,
        noise_type='uniform',
        target_type='coordinate',
        device="cpu",

        use_ot_train: bool = False,
        ot_method:    str  = "sinkhorn",
        ot_reg:       float= 0.05,
    ):
        self.device          = device
        self.encoded_dim     = encoded_dim
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.val_split       = val_split
        self.signal_strength = signal_strength
        self.noise_type      = noise_type
        self.target_type     = target_type

        # OT logic
        self.use_ot_train = use_ot_train
        if self.use_ot_train:
            self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg)

        ds = SSPDataset(
            data_dir        = data_dir,
            ssp_dim         = self.encoded_dim,
            target_type     = self.target_type,
            noise_type      = self.noise_type,
            signal_strength = self.signal_strength
        )
        self.train_dataset, self.val_dataset = ds.split_dataset(self.val_split)
        self.model = architecture.to(self.device)

    def validate(self, dataloader):
        """Validate using *exact* OT pairing when enabled."""
        self.model.eval()
        total = 0.0
        criterion = nn.CosineEmbeddingLoss()

        with torch.no_grad():
            for batch in dataloader:
                z0_all = batch[0].squeeze(1).to(self.device)
                z1_all = batch[1].squeeze(1).to(self.device)

                # use OT if requested
                if self.use_ot_train:
                    z0, z1 = self.ot_sampler.hard_pair(z0_all, z1_all)
                else:
                    z0, z1 = z0_all, z1_all

                pred = self.model(z0)
                total += criterion(
                    pred, z1,
                    torch.ones(pred.size(0), device=self.device)
                ).item()

        return total / len(dataloader)

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  prefetch_factor=2,)
        
        val_loader   = DataLoader(self.val_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True, 
                                  prefetch_factor=2,)

        loss_curve, val_loss_curve = [], []
        criterion = nn.CosineEmbeddingLoss()
        # criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                z0_all = batch[0].squeeze(1).to(self.device)
                z1_all = batch[1].squeeze(1).to(self.device)

                # OT pairing if enabled
                if self.use_ot_train:
                    # z0, z1 = self.ot_sampler.sample_plan(z0_all, z1_all, replace=False)
                    z0, z1 = self.ot_sampler.hard_pair(z0_all, z1_all)
                else:
                    z0, z1 = z0_all, z1_all

                pred = self.model(z0)
                loss = criterion(pred, z1, torch.ones(pred.size(0), device=self.device))
                # loss = criterion(pred, z1)
                loss.backward()
                optimizer.step()
                total += loss.item()

            avg_train = total / len(train_loader)
            avg_val   = self.validate(val_loader)

            loss_curve.append(avg_train)
            val_loss_curve.append(avg_val)
            print(f"Epoch {epoch+1}/{self.epochs}, Train: {avg_train:.4f}, Val: {avg_val:.4f}")

        return self.model, loss_curve, val_loss_curve
