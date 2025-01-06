import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from cleanup_ssps import sspspace 
from cleanup_ssps.model import MLP
from cleanup_ssps.cleanup_methods import RectifiedFlow
from cleanup_ssps.dataset import SSPDataset
import sys

class RectifiedFlowTrainer:
    def __init__(self, encoded_dim, architecture, data_dir, batch_size=250, epochs=75, lr=1e-4, weight_decay=1e-4, val_split=0.1, signal_strength=0, noise_type='uniform_hypersphere', target_type='coordinate', logit_m=-1.0, logit_s=2.):
        self.encoded_dim = encoded_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_split = val_split
        self.signal_strength = signal_strength
        self.noise_type = noise_type
        self.target_type = target_type
        self.logit_m = logit_m
        self.logit_s = logit_s

        self.rectified_flow = RectifiedFlow(model=architecture, num_steps=100, logit_m=self.logit_m, logit_s=self.logit_s)

        self.dataset = SSPDataset(
            data_dir=data_dir,
            ssp_dim=self.encoded_dim,
            target_type=self.target_type,
            noise_type=self.noise_type,
            signal_strength=self.signal_strength
        )

        self.train_dataset, self.val_dataset = self.dataset.split_dataset(self.val_split)
    
    def validate_flow_prediction(self, dataloader):
        self.rectified_flow.model.eval()
        total_loss = 0
        criterion = nn.CosineEmbeddingLoss()

        with torch.no_grad():
            for batch in dataloader:
                z_init, z1 = batch[0], batch[1]
                z_init, z1 = z_init.squeeze(1), z1.squeeze(1)
                z_t, t, target = self.rectified_flow.get_train_tuple(z_init=z_init, z1=z1)
                pred = self.rectified_flow.model(z_t, t)
                loss = criterion(pred, target, torch.ones(pred.shape[0]))
                total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def validate_cleanup(self, dataloader, N=1):
        self.rectified_flow.model.eval()
        total_loss = 0
        criterion = nn.CosineEmbeddingLoss()

        with torch.no_grad():
            for batch in dataloader:
                z_init, z1 = batch[0], batch[1]
                z_init, z1 = z_init.squeeze(1), z1.squeeze(1)
                outputs = self.rectified_flow.sample_ode(z_init=z_init, N=N)[-1]
                loss = criterion(outputs, z1, torch.ones(outputs.shape[0], device=outputs.device))
                total_loss += loss.item()

        return total_loss / len(dataloader)

    
    def validate(self, dataloader, mode='cleanup', N=1):
        if mode == 'flow_prediction':
            return self.validate_flow_prediction(dataloader)
        elif mode == 'cleanup':
            return self.validate_cleanup(dataloader, N=N)
        else:
            raise ValueError(f"Unknown validation mode: {mode}")


    def train(self):
        optimizer = torch.optim.Adam(self.rectified_flow.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        loss_curve = []
        val_loss_curve = []
        criterion = nn.CosineEmbeddingLoss()

        for epoch in tqdm(range(self.epochs), desc='Training Progress', file=sys.stdout):
            self.rectified_flow.model.train()
            epoch_loss = 0

            for batch in train_dataloader:
                optimizer.zero_grad()
                z0, z1 = batch[0], batch[1]
                z0, z1 = z0.squeeze(1), z1.squeeze(1)
                z_t, t, target = self.rectified_flow.get_train_tuple(z0=z0, z1=z1)
                pred = self.rectified_flow.model(z_t, t)
                loss = criterion(pred, target, torch.ones(pred.shape[0]))

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # scheduler.step()

            avg_train_loss = epoch_loss / len(train_dataloader)
            avg_val_loss = self.validate(val_dataloader)

            loss_curve.append(avg_train_loss)
            val_loss_curve.append(avg_val_loss)

            tqdm.write(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4e}, Val Loss: {avg_val_loss:.4e}')

        return self.rectified_flow.model, loss_curve, val_loss_curve


class FeedforwardTrainer:
    def __init__(self, encoded_dim, architecture, data_dir, batch_size=250, epochs=100, lr=5e-4, weight_decay=1e-4, val_split=0.1, signal_strength=0, noise_type='uniform', target_type='coordinate'):
        self.encoded_dim = encoded_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.signal_strength = signal_strength
        self.noise_type = noise_type
        self.target_type = target_type

        self.dataset = SSPDataset(
            data_dir=data_dir,
            ssp_dim=self.encoded_dim,
            target_type=self.target_type,
            noise_type=self.noise_type,
            signal_strength =self.signal_strength
        )

        self.train_dataset, self.val_dataset = self.dataset.split_dataset(self.val_split)

        self.model = architecture


    def interpolate(self, z0, z1):
        t = torch.rand((z1.shape[0], 1), device=z0.device)
        z_t = t * z1 + (1 - t) * z0
        return z_t

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        criterion = nn.CosineEmbeddingLoss()

        with torch.no_grad():
            for batch in dataloader:
                z0, z1 = batch[0], batch[1]
                z0, z1 = z0.squeeze(1), z1.squeeze(1)

                pred = self.model(z0)
                loss = criterion(pred, z1, torch.ones(pred.shape[0], device=pred.device))

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        loss_curve = []
        val_loss_curve = []

        criterion = nn.CosineEmbeddingLoss()

        for epoch in tqdm(range(self.epochs), desc='Training Progress'):
            self.model.train()
            epoch_loss = 0

            for batch in train_dataloader:
                optimizer.zero_grad()
                z0, z1 = batch[0], batch[1]
                z0, z1 = z0.squeeze(1), z1.squeeze(1)

                pred = self.model(z0)
                loss = criterion(pred, z1, torch.ones(pred.shape[0], device=pred.device))

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_dataloader)
            avg_val_loss = self.validate(val_dataloader)

            loss_curve.append(avg_train_loss)
            val_loss_curve.append(avg_val_loss)

            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        return self.model, loss_curve, val_loss_curve
