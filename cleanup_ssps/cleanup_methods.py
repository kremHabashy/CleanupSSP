import torch
from cleanup_ssps import sspspace
import numpy as np
from scipy import integrate

class RectifiedFlow():
    def __init__(self, model=None, num_steps=1000, logit_m=0.0, logit_s=1.0):
        self.model = model
        self.N = num_steps
        self.logit_m = logit_m  # Location parameter for logit-normal
        self.logit_s = logit_s  # Scale parameter for logit-normal

    def get_train_tuple(self, z0=None, z1=None):
        # Logit-normal sampling for t
        u = torch.normal(mean=self.logit_m, std=self.logit_s, size=(z1.shape[0], 1)).to(z1.device)
        t = torch.sigmoid(u)  # Apply logistic function to get t in range (0, 1)

        z_t = t * z1 + (1.0 - t) * z0
        target = z1 - z0
        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z_init=None, N=None, reverse=False):
        if N is None:
            N = self.N

        if reverse:
            dt = -1.0 / N
            time_steps = torch.linspace(1.0, 0.0, N + 1)
        else:
            dt = 1.0 / N
            time_steps = torch.linspace(0.0, 1.0, N + 1)

        traj = []
        z = z_init.detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone())

        for i in range(N):
            t = torch.ones((batchsize, 1)) * time_steps[i]
            pred = self.model(z, t)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        return traj

    def sample_ode_generative_bbox(self, z_init=None, solver='RK45', eps=1e-3, rtol=1e-5, atol=1e-5):
        dshape = z_init.shape
        device = z_init.device
        traj = []  # List to store the trajectory

        def ode_func(t, x):
            x = torch.from_numpy(x.reshape(dshape)).to(device).float()
            
            # Expand t to match the shape of x
            vec_t = torch.ones(dshape[0], device=x.device) * t
            vec_t = vec_t.unsqueeze(1)  # Expand to match x_input's second dimension
            
            # Now pass to the model
            vt = self.model(x, vec_t)
            vt = vt.detach().cpu().numpy().reshape(-1)
            
            # Append current state to trajectory (if you want to track intermediate steps)
            traj.append(x.detach().clone().cpu())
            return vt

        # Solve the ODE using the specified solver
        solution = integrate.solve_ivp(
            ode_func, (1, eps), z_init.detach().cpu().numpy().reshape(-1),
            method=solver, rtol=rtol, atol=atol
        )
        
        # Number of function evaluations
        nfe = solution.nfev
        
        # Final result (at the last time step)
        result = torch.from_numpy(solution.y[:, -1].reshape(dshape)).to(device)
        
        return result, traj, nfe

  
"""
NOTE: The autoencoder method uses the same MLP, and has to reconstruct the SSP. 
It does not need to be defined here since it is effectively a sub-component of the rectified flow
(i.e. a RF is a multistep autoencoder)
"""

# Effectively, parameter count is simply the number of SSPs we are computing the dot product with.
class DotProductCleanup:
  def __init__(self, num_points=200):
    self.num_points = num_points
    self.ssp_space = sspspace()

  def clean_up(ssp):
    sample_ssps = self.ssp_space.get_sample_ssps(self.num_points)
    sims = sample_ssps.T @ ssp
    return sample_ssps[:,np.argmax(sims)]