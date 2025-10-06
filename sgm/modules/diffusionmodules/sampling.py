"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import numpy as np
import os
from PIL import Image
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (get_ancestral_step,
                                                        linear_multistep_coeff,
                                                        to_d, to_neg_log_sigma,
                                                        to_sigma)
from ...util import append_dims, default, instantiate_from_config

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def prepare_sampling_loop2(self, x, cond_a, cond_b, uc_a=None, uc_b=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc_a = default(uc_a, cond_a)
        uc_b = default(uc_b, cond_b)


        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond_a, cond_b, uc_a, uc_b
    
    
    def denoise(self, x, denoiser, sigma, cond, uc, is_front_view, kv_f_list):
        if is_front_view:
            denoised, kv_f_list = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), is_front_view, kv_f_list)
            denoised = self.guider(denoised, sigma)
            return denoised, kv_f_list
        else:
            denoised, kv_f_list = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), is_front_view, kv_f_list)
            denoised = self.guider(denoised, sigma)
            return denoised, kv_f_list


    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        kv_f_list = []
        is_front_view = True
        denoised, _ = self.denoise(x, denoiser, sigma_hat, cond, uc, is_front_view, kv_f_list)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            print("i:", i)
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step

class DualConditionEDMSampler(EulerEDMSampler):  
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize alpha_t with a custom pattern
        # alpha_start = np.array([1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0, 0.0,
        #                         0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0])
        # alpha_end = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1,
        #                         0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0])
        
        alpha_start = np.array([1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0, 0.0,
                                0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0])
        alpha_end = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        
        # indices = np.arange(21)
        # distances = np.minimum(indices, 21 - indices)
        # sigma_start = 5.0
        # sigma_end = 3.5
        # alpha_start = np.exp(- (distances ** 2) / (2 * sigma_start ** 2))
        # alpha_end = np.exp(- (distances ** 2) / (2 * sigma_end ** 2))
        # alpha_start = np.round(alpha_start, 1)
        # alpha_end = np.round(alpha_end, 1)

        n_steps = 50
        alpha_t = np.linspace(alpha_start, alpha_end, 35)  
        alpha_t = np.concatenate([alpha_t, np.tile(alpha_end, (n_steps - 35, 1))]) 
        self.alpha_t = torch.tensor(alpha_t, dtype=torch.float32)
        
        self.sobel_x = (torch.tensor([[ -1., 0., 1.],
                                    [ -2., 0., 2.],
                                    [ -1., 0., 1.]])* 0.1).view(1, 1, 3, 3)

        self.sobel_y = (torch.tensor([[ 1.,  2.,  1.],
                                    [ 0.,  0.,  0.],
                                    [-1., -2., -1.]])* 0.1).view(1, 1, 3, 3)

        self.high_pass = (torch.tensor([[0., -1., 0.],
                                    [-1., 4., -1.],
                                    [0., -1., 0.]])* 0.1).view(1, 1, 3, 3)
        
        
    def reorder_output(self, output, p):
        reordered_output = []
    
        for tensor_pair in output: 
            reordered_pair = []     
            
            for tensor in tensor_pair:  
                front_half = tensor[:21]
                back_half = tensor[21:]
                
                reordered_front_half = torch.zeros_like(front_half)
                reordered_back_half = torch.zeros_like(back_half)
                
                for i in range(21):
                    reordered_front_half[i] = front_half[(i + p - 1) % 21]
                    reordered_back_half[i] = back_half[(i + p - 1) % 21]
                
                reordered_tensor = torch.cat([reordered_front_half, reordered_back_half], dim=0)
                reordered_pair.append(reordered_tensor)
            
            reordered_output.append(tuple(reordered_pair))  
        
        return reordered_output
    
    def _fuse_denoised(self, denoised, step_index):
        transition_steps = 10 
        if step_index < 30 or step_index >= 40:
            return denoised
        else:
            progress = (step_index - 30) / transition_steps  # Progress from 0 to 1
            w1 = 1.0 - 0.2 * progress
            w2 = 0.1 * progress
            w3 = 0.1 * progress
        
        device = denoised.device
        dtype = denoised.dtype
        
        sobel_x = self.sobel_x.repeat(denoised.size(1), 1, 1, 1).to(device=device, dtype=dtype)
        sobel_y = self.sobel_y.repeat(denoised.size(1), 1, 1, 1).to(device=device, dtype=dtype)
        high_pass = self.high_pass.repeat(denoised.size(1), 1, 1, 1).to(device=device, dtype=dtype)
        
        grad_x = F.conv2d(denoised, sobel_x, padding=1, groups=denoised.size(1))
        grad_y = F.conv2d(denoised, sobel_y, padding=1, groups=denoised.size(1))
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        textures = F.conv2d(denoised, high_pass, padding=1, groups=denoised.size(1))
        fused = w1 * denoised + w2 * edges + w3 * textures  
        
        return fused
    
    
        
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond_f, cond_b, path_b_num, step_index, uc_f=None, uc_b=None, gamma=0.0, if_use_mf=False):
        sigma_hat = sigma * (gamma + 1.0)
        p = path_b_num
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        
        x_reordered = torch.zeros_like(x)
        
        for i in range(21):
            x_reordered[i] = x[(i + p - 1) % 21]
        
        if step_index < 20:
            kv_f_list = []
            is_front_view = True
            denoised_a, kv_f_list = self.denoise(x, denoiser, sigma_hat, cond_f, uc_f, is_front_view, kv_f_list)
            reordered_kv_f_list = self.reorder_output(kv_f_list, p)
            is_front_view = False
            denoised_b, _ = self.denoise(x_reordered, denoiser, sigma_hat, cond_b, uc_b, is_front_view, reordered_kv_f_list)
        else:
            kv_f_list = []
            is_front_view = True
            denoised_a, _ = self.denoise(x, denoiser, sigma_hat, cond_f, uc_f, is_front_view, kv_f_list)
            denoised_b, _ = self.denoise(x_reordered, denoiser, sigma_hat, cond_b, uc_b, is_front_view, kv_f_list)
        
        
        alpha = self.alpha_t[step_index]

        denoised_b_reordered = torch.zeros_like(denoised_b)  
        alpha_b_reordered = torch.zeros_like(alpha)  
        
        for i in range(21):
            alpha_b_reordered[i] = alpha[(i+21-p+1)%21]  
            denoised_b_reordered[i] = denoised_b[(i+21-p+1)%21]  
        
        denoised = torch.zeros_like(denoised_a)  

        for i in range(21):
            denoised[i] = (1 - alpha_b_reordered[i]) * denoised_a[i] + alpha_b_reordered[i] * denoised_b_reordered[i]
        
        
        if if_use_mf:
            denoised = self._fuse_denoised(denoised, step_index)

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond_f, uc_f
        )
        return x

   
    def __call__(self, denoiser, x, cond_f, cond_b, path_b_num, uc_f=None, uc_b=None, num_steps=None, if_use_mf=False):
        x, s_in, sigmas, num_sigmas, cond_f, cond_b, uc_f, uc_b = self.prepare_sampling_loop2(
            x, cond_f, cond_b, uc_f, uc_b, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            print("i:", i)
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond_f,  
                cond_b,
                path_b_num,  
                i,  
                uc_f,    
                uc_b,    
                gamma,
                if_use_mf=if_use_mf,
            )

        return x
    

class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x
