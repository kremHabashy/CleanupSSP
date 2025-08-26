import math
import torch
from scipy import integrate
import torch.nn.functional as F
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

class FlowMatching:
    def __init__(
        self,
        model,
        num_steps: int = 1000,
        sampling: str = "deterministic",
        sigma_min: float = 0.1,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        device: str = "cpu"
    ):
        """
        model      – your torch.nn.Module
        num_steps  – number of integration steps if you do ODE sampling
        sampling   – one of "deterministic", "improved_fm", "schrodinger", "vp_diffusion"
        sigma_min  – σ_min for both improved_fm and VE diffusion
        sigma_max  – σ_max for VE diffusion
        beta_min   – β_min for VP diffusion
        beta_max   – β_max for VP diffusion
        device     – cpu / cuda
        """
        self.model       = model
        self.N           = num_steps
        self.sampling    = sampling
        self.sigma_min   = sigma_min
        self.beta_min    = beta_min
        self.beta_max    = beta_max
        self.device      = device

    def get_train_tuple(self, z0, z1):
        """
        Returns:
          z_t    – the perturbed/interpolated state at time t
          t      – (B,1) times
          u_true – the ground‐truth vector field at that (z_t, t)
        """
        B = z0.shape[0]
        z0, z1 = z0.to(self.device), z1.to(self.device)

        # draw t ∈ (0,1)
        eps = 1e-4
        t = torch.rand((B,1), device=self.device) * (1 - 2*eps) + eps

        if self.sampling == "hyperspherical_fm":
            v = self.logmap_sphere(z0, z1)          # tangent at z0
            z_t = self.expmap_sphere(z0, t * v)     # geodesic at time t
            u_true = v                              # constant true velocity

        else:
            # Euclidean-based sampling
            mean = t * z1 + (1.0 - t) * z0

            if self.sampling == "deterministic":
                z_t    = mean
                u_true = (z1 - z0)

            elif self.sampling == "improved_fm":
                z_t    = torch.normal(mean, self.sigma_min)
                u_true = (z1 - z0)

            elif self.sampling == "schrodinger":
                var    = t * (1.0 - t) * (self.sigma_min ** 2)
                var    = var.clamp(min=1e-6)
                
                std    = torch.sqrt(var)
                z_t    = torch.normal(mean, std)
                denom  = 2.0 * t * (1.0 - t)
                denom  = denom.clamp(min=1e-6)
                corr   = (z_t - mean) / denom
                u_true = (1.0 - 2.0 * t) * corr + (z1 - z0)

            elif self.sampling == "vp_diffusion":
                T_t   = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
                α_t   = torch.exp(-0.5 * T_t)
                var_t = (1.0 - α_t**2).clamp(min=1e-6)
                std   = torch.sqrt(var_t)
                z_t   = torch.normal(mean=α_t * z1, std=std)

                dα_dt = -0.5 * (self.beta_min + (self.beta_max - self.beta_min) * t) * α_t
                u_true = (dα_dt / var_t) * (α_t * z_t - z1)

            else:
                raise ValueError(f"Unknown sampling mode: {self.sampling!r}")

        return z_t, t, u_true
    
    @torch.no_grad()
    def sample_ode(
        self,
        z_init,
        N: int,
        reverse: bool = False,
        use_sphere: bool = False,
        t0: torch.Tensor = None,      # optional starting time in [0,1]
    ):
        """
        ODE sampling with optional hyperspherical integration.
        If t0 is provided, we integrate from t0 -> 1.0 (forward only).
        Otherwise we fall back to start=0->1 or 1->0 when reverse=True.
        """
        if N is None:
            N = self.N

        # 1) initialize point
        z = z_init.detach().clone().to(self.device)

        # 2) determine integration interval
        if t0 is None:
            # old behaviour
            if reverse:
                start, end = 1.0, 0.0
            else:
                start, end = 0.0, 1.0
        else:
            # collapse batched t0 to a single scalar
            start = float(t0.mean().item()) if isinstance(t0, torch.Tensor) else float(t0)
            end   = 1.0

        # 3) build time grid
        dt = (end - start) / N
        time_steps = torch.linspace(start, end, N + 1, device=self.device)

        # 4) integrate
        traj = [z.clone()]
        for i in range(N):
            t = time_steps[i]
            t_vec = t.expand(z.shape[0], 1)  # shape (B,1)

            out = self.model(z, t_vec)
            # if your model returns (v_pred, t_hat), unpack:
            v = out[0] if isinstance(out, tuple) else out

            step = v * dt
            if use_sphere or self.sampling == "hyperspherical_fm":
                z = self.expmap_sphere(z, step)
            else:
                z = z + step
                z = z / z.norm(dim=-1, keepdim=True)

            traj.append(z.clone())

        return traj

    @torch.no_grad()
    def sample_sde(self, z_init, N=50, eps_fn=lambda t: 1e-3):
        """
        Euler–Maruyama SDE sampling:
            dX_t = b(t,X_t) dt - (eps/gamma(t))*eta(t,X_t) dt + sqrt(2 eps) dW_t
        Requires that self.denoiser_model has been set.
        """
        assert hasattr(self, "denoiser_model") and self.denoiser_model is not None, \
            "Attach a denoiser_model before calling sample_sde()."

        # trivial 1‐step return
        if N <= 1:
            return z_init.unsqueeze(0)  # shape [1, B, D]

        # avoid t=0 so gamma never zero
        eps = 1e-3
        ts  = torch.linspace(eps, 1.0, N, device=self.device)
        dt  = ts[1] - ts[0]

        z    = z_init.to(self.device)
        traj = [z]

        for i in range(N - 1):
            t     = ts[i].view(1,1).expand(z.size(0),1)  # (B,1)
            out   = self.model(z, t)
            drift = out[0] if isinstance(out, tuple) else out

            # compute gamma(t)
            if self.sampling == "improved_fm":
                gamma = torch.full_like(t, self.sigma_min)
            elif self.sampling == "schrodinger":
                gamma = (torch.sqrt(t * (1 - t)) * self.sigma_min).clamp(min=1e-6)
            elif self.sampling == "vp_diffusion":
                T_t   = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
                alpha = torch.exp(-0.5 * T_t)
                gamma = torch.sqrt(1 - alpha**2).clamp(min=1e-6)
            else:
                raise RuntimeError(f"SDE sampling not supported for mode={self.sampling}")

            eta    = self.denoiser_model(z, t)
            eps_val = eps_fn(t)
            noise  = torch.randn_like(z) * torch.sqrt(2 * eps_val * dt)

            z = z + (drift - (eps_val / gamma) * eta) * dt + noise
            z = z / z.norm(dim=1, keepdim=True)
            traj.append(z)

        return torch.stack(traj, dim=0)  # [N, B, D]


    @staticmethod
    def logmap_sphere(p, q):
        """
        Logarithm map on the hypersphere: tangent vector v at p pointing to q
        """
        dot = (p * q).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        v = q - dot * p
        v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
        return theta * v / v_norm

    @staticmethod
    def expmap_sphere(p, v):
        """
        Exponential map on the hypersphere: move from p along tangent v
        """
        norm_v = torch.norm(v, dim=-1, keepdim=True) + 1e-8
        return torch.cos(norm_v) * p + torch.sin(norm_v) * (v / norm_v)