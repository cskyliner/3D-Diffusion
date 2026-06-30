from __future__ import annotations

from typing import Callable

import torch

from .ddim import make_ddim_sampling_parameters, make_ddim_timesteps

TensorCallback = Callable[[int, int, torch.Tensor], None]


def _maybe_progress(iterable, enabled: bool, total: int | None = None, desc: str = "PLMS"):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


class PLMSSampler:
    def __init__(self, diffusion) -> None:
        self.diffusion = diffusion

    def make_schedule(
        self,
        steps: int,
        ddim_discretize: str = "uniform",
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        device = torch.device(device)
        timesteps = make_ddim_timesteps(ddim_discretize, steps, self.diffusion.num_timesteps, device)
        sigmas, alphas, alphas_prev = make_ddim_sampling_parameters(
            self.diffusion.alphas_cumprod.to(device),
            timesteps,
            eta=0.0,
        )
        return {
            "timesteps": timesteps,
            "sigmas": sigmas,
            "alphas": alphas,
            "alphas_prev": alphas_prev,
            "sqrt_one_minus_alphas": (1.0 - alphas).sqrt(),
        }

    def _get_model_eps(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: object,
        guidance_scale: float,
        unconditional_cond: object,
        score_corrector: Callable | None,
        corrector_kwargs: dict | None,
    ) -> torch.Tensor:
        eps = self.diffusion.apply_model(
            x,
            t,
            cond,
            guidance_scale=guidance_scale,
            unconditional_cond=unconditional_cond,
        )
        if score_corrector is not None:
            corrector_kwargs = corrector_kwargs or {}
            eps = score_corrector.modify_score(self.diffusion, eps, x, t, cond, **corrector_kwargs)
        return eps

    def _get_x_prev_and_pred_x0(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        index: int,
        schedule: dict[str, torch.Tensor],
        clip_denoised: bool = False,
        quantize_denoised: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = schedule["alphas"][index].view(1, *([1] * (x.ndim - 1))).to(x)
        alpha_prev = schedule["alphas_prev"][index].view(1, *([1] * (x.ndim - 1))).to(x)
        sqrt_one_minus_alpha = schedule["sqrt_one_minus_alphas"][index].view(1, *([1] * (x.ndim - 1))).to(x)
        pred_x0 = (x - sqrt_one_minus_alpha * eps) / alpha.sqrt()
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
        if quantize_denoised is not None:
            pred_x0 = quantize_denoised(pred_x0)
        if index == 0:
            return pred_x0, pred_x0
        direction = (1.0 - alpha_prev).sqrt() * eps
        x_prev = alpha_prev.sqrt() * pred_x0 + direction
        return x_prev, pred_x0

    def p_sample_plms(
        self,
        x: torch.Tensor,
        cond: object,
        t: torch.Tensor,
        index: int,
        schedule: dict[str, torch.Tensor],
        old_eps: list[torch.Tensor],
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
        clip_denoised: bool = False,
        quantize_denoised: Callable[[torch.Tensor], torch.Tensor] | None = None,
        score_corrector: Callable | None = None,
        corrector_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = self._get_model_eps(x, t, cond, guidance_scale, unconditional_cond, score_corrector, corrector_kwargs)
        if index == 0:
            x_prev, pred_x0 = self._get_x_prev_and_pred_x0(x, eps, index, schedule, clip_denoised, quantize_denoised)
            return x_prev, pred_x0, eps

        if len(old_eps) == 0:
            x_euler, _ = self._get_x_prev_and_pred_x0(x, eps, index, schedule, clip_denoised, quantize_denoised)
            next_timestep = schedule["timesteps"][index - 1]
            next_t = torch.full((x.shape[0],), int(next_timestep.item()), device=x.device, dtype=torch.long)
            eps_next = self._get_model_eps(
                x_euler,
                next_t,
                cond,
                guidance_scale,
                unconditional_cond,
                score_corrector,
                corrector_kwargs,
            )
            eps_prime = (eps + eps_next) / 2.0
        elif len(old_eps) == 1:
            eps_prime = (3.0 * eps - old_eps[-1]) / 2.0
        elif len(old_eps) == 2:
            eps_prime = (23.0 * eps - 16.0 * old_eps[-1] + 5.0 * old_eps[-2]) / 12.0
        else:
            eps_prime = (55.0 * eps - 59.0 * old_eps[-1] + 37.0 * old_eps[-2] - 9.0 * old_eps[-3]) / 24.0

        x_prev, pred_x0 = self._get_x_prev_and_pred_x0(
            x,
            eps_prime,
            index,
            schedule,
            clip_denoised=clip_denoised,
            quantize_denoised=quantize_denoised,
        )
        return x_prev, pred_x0, eps

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        steps: int = 100,
        eta: float = 0.0,
        cond: object = None,
        guidance_scale: float = 1.0,
        unconditional_cond: object = None,
        device: torch.device | str = "cpu",
        ddim_discretize: str = "uniform",
        x_T: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        clip_denoised: bool = False,
        quantize_denoised: Callable[[torch.Tensor], torch.Tensor] | None = None,
        score_corrector: Callable | None = None,
        corrector_kwargs: dict | None = None,
        return_intermediates: bool = False,
        log_every_t: int = 100,
        callback: TensorCallback | None = None,
        img_callback: TensorCallback | None = None,
        progress: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        if eta != 0.0:
            raise ValueError("PLMS sampler only supports eta=0 in this implementation.")
        diffusion = self.diffusion
        device = torch.device(device)
        schedule = self.make_schedule(steps, ddim_discretize=ddim_discretize, device=device)
        timesteps = schedule["timesteps"].flip(0)
        x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
        old_eps: list[torch.Tensor] = []
        intermediates: dict[str, list[torch.Tensor]] = {"x_inter": [x.detach().cpu()], "pred_x0": []}
        iterator = _maybe_progress(list(enumerate(timesteps)), progress, total=len(timesteps), desc="PLMS")
        for loop_index, step in iterator:
            schedule_index = len(timesteps) - loop_index - 1
            t = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
            if mask is not None and x0 is not None:
                img_orig = diffusion.q_sample(x0.to(device), t)
                x = img_orig * mask.to(device) + (1.0 - mask.to(device)) * x
            x, pred_x0, eps = self.p_sample_plms(
                x,
                cond,
                t,
                schedule_index,
                schedule,
                old_eps,
                guidance_scale=guidance_scale,
                unconditional_cond=unconditional_cond,
                clip_denoised=clip_denoised,
                quantize_denoised=quantize_denoised,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            old_eps.append(eps.detach())
            old_eps = old_eps[-3:]
            if callback is not None:
                callback(loop_index, int(step.item()), x)
            if img_callback is not None:
                img_callback(loop_index, int(step.item()), pred_x0)
            if return_intermediates and (loop_index % log_every_t == 0 or loop_index == len(timesteps) - 1):
                intermediates["x_inter"].append(x.detach().cpu())
                intermediates["pred_x0"].append(pred_x0.detach().cpu())
        if return_intermediates:
            return x, intermediates
        return x
