
from tqdm.auto import trange

@torch.no_grad()
def sample_dpmpp_2m_alt(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        sigma_progress = i / len(sigmas) 
        adjustment_factor = 1 + (0.15 * (sigma_progress * sigma_progress)) 
        old_denoised = denoised * adjustment_factor
    return x

k_diffusion.sampling.sample_dpmpp_2m_alt = sample_dpmpp_2m_alt

samplers_data_k_diffusion.insert(17, sd_samplers_common.SamplerData('DPM++ 2M importos', lambda model: KDiffusionSampler('sample_dpmpp_2m_alt', model), ['k_dpmpp_2m_alt'], {}))
samplers_data_k_diffusion.insert(18, sd_samplers_common.SamplerData('DPM++ 2M importos Karras', lambda model: KDiffusionSampler('sample_dpmpp_2m_alt', model), ['k_dpmpp_2m_alt_ka'], {'scheduler': 'karras'}))
