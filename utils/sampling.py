from scipy import integrate
import numpy as np
import abc
import torch

import model.sde_lib as sde_lib

import sys

def get_sampling_fn(world, sde):
    """Create a sampling function.

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers representing the expected shape of a single sample.
        inverse_scaler: The inverse data normalizer function.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """
    if world.config['sampling_method'].lower() == 'pc':
        predictor_fn = get_predictor(world.config['predictor'])
        corrector_fn = get_corrector(world.config['corrector'])
        sampling_fn = get_pc_sampler(sde,
                                     predictor_fn,
                                     corrector_fn,
                                     world.config['snr'],
                                     world.config['sampling_scale'],
                                     world.args.continuous)
        
        
        # predictor = get_predictor(config.sampling.predictor.lower())

        
    else:
        raise ValueError(f"Sampler name {world.config['sampling_method']} unknown.")

    return sampling_fn
  
  
def get_score_fn(sde, model, continuous):
    if isinstance(sde, sde_lib.VPSDE):
        def score_fn(x, t, c):
            # Scale neural network output by standard deviation and flip sign
            if continuous: 
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                
                labels = t
                score = model(x, labels, c)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
                # score = -score / std[:, None]  # ----------------------------------- or 2

            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model(x, labels, c)
                std = sde.sqrt_1m_alphas_cumprod.to(x.device)[labels.long()]
                score = -score / std[:, None]  
            return score 
    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t, c):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

                score = model(x, labels, c)
            return score
    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    return score_fn          
      


def get_pc_sampler(sde, predictor_fn, corrector_fn, snr, 
                   sampling_scale, continuous,
                   n_steps=1,
                   eps=1e-3,  # eps=1e-3 right,  light: lGN
                   start_from_noise = False, 
                   denoise=True):     
    
    def pc_sampler(model, x_start, c):

        score_fn = get_score_fn(sde, model, continuous)
        predictor = predictor_fn(sde, score_fn)
        corrector = corrector_fn(sde, score_fn, snr, n_steps)
        with torch.no_grad():
            # initial sample
            if start_from_noise:
                x = sde.prior_sampling(x_start.shape).to(x_start.device) 
            else:
                if continuous:
                    t = torch.tensor([sde.T] * x_start.shape[0]).to(x_start.device)
                    z = torch.randn_like(x_start)
                    mean, std = sde.marginal_prob(x_start, t)
                    x = mean + std[:, None] * z
                    del t
                    del z
                    del mean, std

                else:  
                    if isinstance(sde, sde_lib.VPSDE):
                        labels = torch.tensor([sampling_scale-1] * x_start.shape[0]).to(x_start.device)
                        sqrt_alphas_cumprod = sde.sqrt_alphas_cumprod.to(x_start.device)
                        sqrt_1m_alphas_cumprod = sde.sqrt_1m_alphas_cumprod.to(x_start.device)
                        noise = torch.randn_like(x_start)
                        x = sqrt_alphas_cumprod[labels, None] * x_start + \
                            sqrt_1m_alphas_cumprod[labels, None] * noise
                        del labels
                        del sqrt_alphas_cumprod
                        del sqrt_1m_alphas_cumprod
                    else:
                        labels = torch.tensor([sampling_scale-1] * x_start.shape[0]).to(x_start.device)
                        smld_sigma_array = torch.flip(sde.discrete_sigmas, dims=(0,))
                        sigmas = smld_sigma_array.to(x_start.device)[labels]
                        noise = torch.randn_like(x_start) * sigmas[:, None]
                        x = noise + x_start
                        del labels
                        del noise
                        del sigmas

            # print(x.shape)
            timesteps = torch.linspace(sde.T, eps, sde.N, device= x.device)  
            # a tensor with sde.N values, ranging from sde.T to eps

            # reverse diffusion process
            for i in range(sde.N): 
                t = timesteps[i]
                vec_t = torch.ones(x_start.shape[0], device=t.device) * t
                
                x, x_mean = corrector.update_fn(x, c, vec_t)
                x, x_mean = predictor.update_fn(x, c, vec_t)
        del timesteps
        return (x_mean if denoise else x), sde.N * (n_steps + 1)
    return pc_sampler



def get_predictor(name):
    if name == 'EulerMaruyama':
        return EulerMaruyamaPredictor
    elif name == 'Reverse':
        return ReverseDiffusionPredictor
    elif name == 'Ancestral':
        return AncestralSamplingPredictor

 
def get_corrector(name):
    if name == 'Langevin':
        return LangevinCorrector
    elif name == 'AnnealedLangevin':
        return AnnealedLangevinDynamics
    elif name == 'none':
        return NoneCorrector


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""
    def __init__(self, sde, score_fn, snr, n_steps, scale_eps = 0.1):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        pass

class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, c, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t, c)
        x_mean = x + drift * dt
        # x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        x = x_mean + diffusion[:, None] * np.sqrt(-dt) * z
        return x, x_mean

class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, c, t):
        f, G = self.rsde.discretize(x, t, c)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None] * z
        return x, x_mean

class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE): # and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vpsde_update_fn(self, x, c, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t, c)
        x_mean = (x + beta[:, None] * score) / torch.sqrt(1. - beta)[:, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None] * noise
        return x, x_mean
    
    def vesde_update_fn(self, x, c, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t, c)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None] * noise
        return x, x_mean


    def update_fn(self, x, c, t):
        # if isinstance(self.sde, sde_lib.VESDE):
        #     return self.vesde_update_fn(x, t)
        if isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, c, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, c, t)


class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE)\
            and not isinstance(sde, sde_lib.VESDE) :
            #and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, c, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE): #  or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t, c)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            # x_mean = x + step_size[:, None, None, None] * grad
            # x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
            x_mean = x + step_size[:, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None] * noise

        return x, x_mean

class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, c, t):
        return x, x

class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """


    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
            and not isinstance(sde, sde_lib.VESDE) :
        # and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, c, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

            std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t, c)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None]

            return x, x_mean





