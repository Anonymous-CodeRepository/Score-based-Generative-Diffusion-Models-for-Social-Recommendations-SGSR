from model.sde_lib import VPSDE, VESDE
import torch

import sys

def get_smld_loss_fn(vesde, train, reduce_mean=False):
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."
    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    
    def loss_fn(model, batch, c, labels):
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None]
        perturbed_data = noise + batch
        
        score = model(perturbed_data, labels, c)  
        target = -noise / (sigmas ** 2)[:, None]
        
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        
        return losses
    return loss_fn

def get_ddpm_loss_fn(vpsde, reduce_mean=True):
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)  # whether reduce mean ------------

    def loss_fn(model, batch, c, labels):
        # labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)

        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None] * batch + \
                         sqrt_1m_alphas_cumprod[labels, None] * noise

        score = model(perturbed_data, labels, c)
    
        # Likelihood weighting is not supported for original SMLD/DDPM training
        # losses = torch.square(score + noise / sqrt_1m_alphas_cumprod[labels, None])
        losses = torch.square(score - noise)  # ATTENTION   #########################   ATTENTION #########################################
        
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        # loss = torch.mean(losses* weights)
        
        # pred_xstart = vpsde._predict_xstart_from_eps(x_t=perturbed_data, t=labels, noise=score)

        del labels
        del sqrt_alphas_cumprod
        del sqrt_1m_alphas_cumprod
        del noise
    
        # return losses, pred_xstart
        return losses

    return loss_fn

def get_sde_loss_fn(sde, reduce_mean, likelihood_weighting, eps=1e-5):
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    def loss_fn(model, batch, c, labels):
        # range in (sde.T, eps)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps  
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None] * z
        score = model(perturbed_data, t, c)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)
            
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
            loss = torch.mean(losses)
        
        # pred_xstart = sde._predict_xstart_from_eps_c(x_t=perturbed_data, t=t, score=score)
        del t
        del z
        del perturbed_data
        del mean,std
        # return loss, pred_xstart
        return loss
        
    return loss_fn
        
        
def get_loss_fn(sde, reduce_mean, continuous=False, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        optimize_fn: An optimization function.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
        A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, reduce_mean, likelihood_weighting)
        
    else:
        # not continuous
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, reduce_mean)
        elif isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, reduce_mean)
        else:
            print(type(sde))
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    return loss_fn