import torch
import torch.nn as nn
import numpy as np
from diffusers import DDPMScheduler

def normalize(x: torch.Tensor, dim=None, eps=1e-4) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32) # type: torch.Tensor
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class FourierFeatureExtractor(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32) # type: torch.Tensor
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

class NormalizedLinearLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x: torch.Tensor, gain=1) -> torch.Tensor:
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # type: torch.Tensor # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))
    
class AdaptiveLossWeightMLP(nn.Module):
    def __init__(
            self,
            noise_scheduler: DDPMScheduler,
            logvar_channels=128,
            lambda_weights: torch.Tensor = None,
            importance_weights: torch.Tensor = None,
        ):
        super().__init__()
        self.alphas_cumprod = noise_scheduler.alphas_cumprod.cuda()
        #self.a_bar_mean = noise_scheduler.alphas_cumprod.mean()
        #self.a_bar_std = noise_scheduler.alphas_cumprod.std()
        self.a_bar_mean = self.alphas_cumprod.mean()
        self.a_bar_std = self.alphas_cumprod.std()
        self.logvar_fourier = FourierFeatureExtractor(logvar_channels)
        self.logvar_linear = NormalizedLinearLayer(logvar_channels, 1, kernel=[]) # kernel = []? (not in code given, added matching edm2)
        # create starting weights given tensor of weight values, otherwise flat start
        self.lambda_weights = lambda_weights.cuda() if lambda_weights is not None else torch.ones(1000, device='cuda')
        self.importance_weights = importance_weights.cuda() if importance_weights is not None else torch.ones(1000, device='cuda')

        # # min snr importance weights
        # all_timesteps = torch.arange(noise_scheduler.config.num_train_timesteps, device='cuda')
        # snr = torch.stack([noise_scheduler.all_snr[t] for t in all_timesteps])
        # min_snr_gamma = 20 * torch.minimum(snr, torch.full_like(snr, 1))
        # min_snr_gamma = torch.div(min_snr_gamma, snr + 1).float().cuda()
        # self.importance_weights = torch.where(
        #     self.importance_weights > min_snr_gamma,
        #     self.importance_weights,
        #     min_snr_gamma,
        # )

        self.noise_scheduler = noise_scheduler

    def _forward(self, timesteps: torch.Tensor):
        #a_bar = self.noise_scheduler.alphas_cumprod[timesteps]
        a_bar = self.alphas_cumprod[timesteps]
        c_noise = a_bar.sub(self.a_bar_mean).div_(self.a_bar_std)
        return self.logvar_linear(self.logvar_fourier(c_noise)).squeeze()

    def forward(self, loss: torch.Tensor, timesteps):
        adaptive_loss_weights = self._forward(timesteps)
        loss_scaled = loss * (self.lambda_weights[timesteps] / torch.exp(adaptive_loss_weights)) # type: torch.Tensor
        loss = loss_scaled + (self.importance_weights[timesteps] * adaptive_loss_weights) # type: torch.Tensor

        return loss, loss_scaled
    
def create_weight_MLP(noise_scheduler, logvar_channels=128, lambda_weights=None, importance_weights=None):
    print("creating weight MLP")
    lossweightMLP = AdaptiveLossWeightMLP(noise_scheduler, logvar_channels, lambda_weights, importance_weights)
    MLP_optim = torch.optim.AdamW(lossweightMLP.parameters(), lr=1e-2, weight_decay=0)
    return lossweightMLP, MLP_optim