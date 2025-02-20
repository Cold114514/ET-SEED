import torch, numpy as np
torch.manual_seed(3407)
from etseed.utils.se_math import se3
from etseed.utils.data_utils import bezier_curve
from diffusers import DDPMScheduler
from pdb import set_trace as bp
import math
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

class DiffusionScheduler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_steps: int = 100 # 100
        self.beta_1: float = 1e-4
        self.beta_T: float = 0.05
        self.sigma_r: float = 0.05 # 0.2 0.05 0.001 0.0005
        self.sigma_t: float = 0.03 # 0.1 0.03 0.001 0.0003
        self.mode = 'squaredcos_cap_v2' # ['linear','cosine','squaredcos_cap_v2']
        self.S = 0.008 # 0.008
       
        device = torch.device("cuda")
        
        if self.mode == 'linear':
            self.betas = torch.linspace(self.beta_1, self.beta_T, steps=self.num_steps)      
        elif self.mode == 'cosine':
            def betas_fn(s):
                T = self.num_steps
                def f(t, T, s):
                    return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
                alphas = [f(0, T, s)]
                for t in range(0, T):
                    alphas.append(f(t, T, s) / alphas[-1])
                betas = [1 - alpha / alphas[0] for alpha in alphas]
                return [min(beta, 0.999) for beta in betas]
            betas = betas_fn(s=self.S)
            self.betas = torch.FloatTensor(betas)
        elif self.mode == 'squaredcos_cap_v2':
            self.betas = betas_for_alpha_bar(self.num_steps)
    
        self.alphas = 1.0 - self.betas
        self.log_alphas = torch.log(self.alphas)
        self.alphas_cumsum = torch.cumsum(self.log_alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.alpha_bars = self.alphas_cumsum.exp().to(device)
        
        self.gamma0 = torch.zeros_like(self.betas).to(device)
        self.gamma1 = torch.zeros_like(self.betas).to(device)
        self.gamma2 = torch.zeros_like(self.betas).to(device)
        
        
        for t in range(1, self.num_steps):  # 2 to T
            alpha_prod_t = self.alpha_bars[t]
            alpha_prod_t_prev = self.alpha_bars[t - 1] if t > 0 else self.one
            self.gamma0[t] = self.betas[t] * torch.sqrt(alpha_prod_t_prev) / (1. - alpha_prod_t)
            self.gamma1[t] = (1. - alpha_prod_t_prev) * torch.sqrt(alpha_prod_t) / (1. - alpha_prod_t)
            self.gamma2[t] = (1. - alpha_prod_t_prev) * self.betas[t] / (1. - alpha_prod_t)


    def set_timesteps(self,num_steps):
        self.num_steps = num_steps
    
    def add_noise(self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
        device):
        
        B = original_samples.shape[0]
        
        # H_T: [1,4,4]
        H_T = torch.eye(4)[None].expand(B, -1, -1).to(device)
        
        alpha_bars = self.alpha_bars[timesteps].to(device)[:, None]  # [B, 1]
 
        # interpolation function F 
        # H_t: [1,2,4,4] 
        # print("(1. - torch.sqrt(alpha_bars))",(se3.exp((1. - torch.sqrt(alpha_bars))).dtype))
        # print("H_T @ torch.inverse(original_samples)",(H_T @ (torch.inverse(original_samples)).to(torch.float32)).dtype)
        # print("se3.log(H_T @ (torch.inverse(original_samples).to(torch.float32)))",se3.log(H_T @ (torch.inverse(original_samples).to(torch.float32))))
        # print("H_t",H_T.dtype)
        # print("torch.inverse(original_samples)",torch.inverse(original_samples).dtype)
        # print("H_T @ torch.inverse(original_samples)",(H_T @ torch.inverse(original_samples)).dtype)
        # exit(0)
        H_t = se3.exp((1. - torch.sqrt(alpha_bars)) * se3.log(H_T @ (torch.inverse(original_samples).to(torch.float32)))) @ original_samples.to(torch.float32)
        
        # add noise
        scale = torch.cat([torch.ones(3) * self.sigma_r, torch.ones(3) * self.sigma_t])[None].to(device)  # [1, 6] 
        # Perturbation 
        # print("torch.sqrt(1. - alpha_bars)",torch.sqrt(1. - alpha_bars).shape)
        # print("scale",scale.shape)
        # print("torch.randn(B, 6)",torch.randn(B, 6).shape)
        # exit(0)
        noise = torch.sqrt(1. - alpha_bars) * scale * torch.randn(B, 6).to(device)  # [B, 6]
            
        # Perturbation * interpolation
        H_noise = se3.exp(noise)
            
        H_t_noise = H_noise @ H_t  # [B, 4, 4]
        
        return H_t_noise, H_noise # [B, 4, 4]
    
    
    def denoise(self,
                model_output, # [B, 4, 4]
                timestep,  
                sample, # (B, pred_horizon, 4, 4)
                device):
        
        # # print('model_output',model_output)
        # # print('timestep',timestep)
        # # print('naction', sample)
        
        # B = model_output.shape[0]
        # Rs_pred = model_output[...,:3,:3]
        # ts_pred = model_output[...,:3,3:]
        
        # # print("Rs_pred",Rs_pred)
        # # print("ts_pred",ts_pred)
        # _delta_H_t = torch.cat([Rs_pred, ts_pred], dim=2)  # [B, 3, 4]
        # delta_H_t = torch.eye(4)[None].expand(B, -1, -1).to(device)  # [B, 4, 4]
        # delta_H_t[:, :3, :] = _delta_H_t
        
        # H_0 = delta_H_t @ sample 
        # # print("H_0.shape",H_0.shape)
        # # exit(0)
        # gamma0 = self.gamma0[timestep]
        # gamma1 = self.gamma1[timestep]
        # # print("gamma0.shape",gamma0.shape)
        # # print("se3.log(H_0)",se3.log(H_0).shape)
        # # print("se3.log(H_0)",se3.log(sample).shape)
        # gamma0 = gamma0.unsqueeze(1)
        # gamma1 = gamma1.unsqueeze(1)
        # sample = se3.exp(gamma0 * se3.log(H_0) + gamma1 * se3.log(sample))
        
        
        
        # alpha_bar = self.alpha_bars[timestep].to(device)
        # alpha_bar_ = self.alpha_bars[timestep-1].to(device)
        # beta = self.betas[timestep].to(device)
        # cc = ((1 - alpha_bar_) / (1.- alpha_bar)) * beta
        # scale = torch.cat([torch.ones(3) * self.sigma_r, torch.ones(3) * self.sigma_t])[None].to(device)  # [1, 6]
        # # print("torch.sqrt(cc)",torch.sqrt(cc).shape)
        # # print("scale",scale.shape)
        # # print("torch.randn(B, 6)",torch.randn(B, 6).shape)
        # # exit(0)
        # # [8, 1] * [1, 6]
        # noise = torch.sqrt(cc).unsqueeze(1) * scale * torch.randn(B, 6).to(device)  # [B, 6]
        # H_noise = se3.exp(noise)
        # sample = H_noise @ sample  # [B, 4, 4]
        
        # return sample, H_0
        timestep = timestep[0].cpu()
        B = model_output.shape[0]
        H_0 = (torch.inverse(model_output) @ sample)
        gamma0 = self.gamma0[timestep].to(device)
        gamma1 = self.gamma1[timestep].to(device)
        sample = se3.exp(gamma0 * se3.log(H_0) + gamma1 * se3.log(sample))
        return sample, H_0
    
    

class DiffusionScheduler_vanilla(torch.nn.Module):

    def __init__(self, num_steps=100, beta_1=1e-4, beta_T=0.05, sigma_r_inv=0.05, sigma_t_inv=0.03,
    sigma_r_equiv=0.2, sigma_t_equiv=0.001,  mode='cosine'):
        super().__init__()
        self.num_steps: int = num_steps
        self.beta_1: float = beta_1
        self.beta_T: float = beta_T
        self.sigma_r_inv: float = sigma_r_inv
        self.sigma_t_inv:float = sigma_t_inv
        self.sigma_r_equiv: float = sigma_r_equiv
        self.sigma_t_equiv:float = sigma_t_equiv
        # self.mode = ["linear", "cosine"]
        self.mode = mode
        self.S = 0.008
        self.betas = torch.zeros([self.num_steps + 1])  # 初始化 betas
        device = torch.device("cuda")
        
        if self.mode == 'linear' or self.mode == 'custom':
            betas = torch.linspace(self.beta_1, self.beta_T, steps=self.num_steps)
            self.betas[1:] = betas     # Padding
        elif self.mode == 'cosine':
            def betas_fn(s):
                T = self.num_steps
                def f(t, T, s):
                    return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
                alphas = [f(0, T, s)]
                for t in range(1, T + 1):
                    alphas.append(f(t, T, s) / alphas[-1])
                betas = [1 - alpha / alphas[0] for alpha in alphas]
                return [min(beta, 0.999) for beta in betas]
            betas = betas_fn(s=self.S)
            self.betas = torch.FloatTensor(betas)
        
        self.alphas = 1 - self.betas
        log_alphas = torch.log(self.alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        self.alpha_bars = log_alphas.exp().to(device)

        self.gamma0 = torch.zeros_like(self.betas)
        self.gamma1 = torch.zeros_like(self.betas)
        self.gamma2 = torch.zeros_like(self.betas)
        
        for t in range(2, self.num_steps + 1):  # 2 to T
            self.gamma0[t] = self.betas[t] * torch.sqrt(self.alpha_bars[t - 1]) / (1. - self.alpha_bars[t])
            self.gamma1[t] = (1. - self.alpha_bars[t - 1]) * torch.sqrt(self.alphas[t]) / (1. - self.alpha_bars[t])
            self.gamma2[t] = (1. - self.alpha_bars[t - 1]) * self.betas[t] / (1. - self.alpha_bars[t])
        if self.mode == 'custom':
            gamma1_control = np.array([[0,0.2],[0.1,0.05],[0.2,0.05],[1,0]])
            gamma2_control = np.array([[0,0.8],[0.1,0.95],[0.2,0.95],[1,1]])
            self.gamma0[2:] = torch.tensor(bezier_curve(gamma1_control, np.linspace(0, 1, self.num_steps-1)))
            self.gamma1[2:] = torch.tensor(bezier_curve(gamma2_control, np.linspace(0, 1, self.num_steps-1)))

    def set_timesteps(self,num_steps):
        self.num_steps = num_steps
    
    def add_noise_inv(self,
        original_samples: torch.FloatTensor, # [1, 1, 4, 4]
        timesteps: torch.IntTensor,
        device):
        batch_size = original_samples.shape[0]
        T_a = original_samples.shape[1]
        original_samples = original_samples.reshape(-1, 4, 4)  # [B * horizon, 4, 4]
        scale = torch.cat([torch.ones(3) * self.sigma_r_inv, torch.ones(3) * self.sigma_t_inv])[None].to(device)  # [1, 6]
        alpha_bars = self.alpha_bars[timesteps].to(device)[:, None]  # [B, 1]
 
        H_T = torch.eye(4)[None].expand(batch_size*T_a, -1, -1).to(device)
        # interpolation function F 
        
        F = se3.exp((1. - torch.sqrt(alpha_bars)) * se3.log(H_T @ torch.inverse(original_samples)))

        # Perturbation 
        noise_origin = torch.randn(batch_size, 6).repeat(T_a, 1, 1).transpose(0, 1).reshape(-1, 6)  # [B * horizon, 6]
        noise = torch.sqrt(1. - alpha_bars) * scale * noise_origin.to(device)  # [B, 6]

        # Perturbation * interpolation
        H_noise = se3.exp(noise) @ F  # [B * horizon, 4, 4]
        H_t_noise = H_noise @ original_samples # [B * horizon, 4, 4]

        #! chenrui: 这里的加噪跟点云配准那篇一样
        return H_t_noise, H_noise

    def add_noise_equiv(self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
        device,):
        batch_size = original_samples.shape[0]
        T_a = original_samples.shape[1]
        original_samples = original_samples.reshape(-1, 4, 4)  # [B * horizon, 4, 4]
        alpha_bars = self.alpha_bars[timesteps].to(device)[:, None]  # [B, 1]

        # Perturbation 
        scale = torch.cat([torch.ones(3) * self.sigma_r_equiv, torch.ones(3) * self.sigma_t_equiv])[None].to(device)  # [1, 6]
        noise_origin = torch.randn(batch_size, 6).repeat(T_a, 1, 1).transpose(0, 1).reshape(-1, 6)  # [B * horizon, 6]

        noise = scale * noise_origin.to(device)  # [B, 6]

        # Perturbation
        # print(noise)
        H_noise = se3.exp(noise) # [B * horizon, 4, 4]
        # print(H_noise)
        H_t_noise = H_noise @ original_samples # [B * horizon, 4, 4]

        #! chenrui: 这里的加噪跟点云配准那篇一样
        return H_t_noise, H_noise

    def denoise(self,
                model_output, 
                timestep,  
                sample, 
                device):
        # print("model_output",model_output.shape, # [1, 4, 4]
        #       "timestep.shape", timestep.shape, # [4, 100, 1]
        #       "sample.shape",sample.shape) # [4, 100, 9]
        # exit(0)
        timestep = timestep[0].cpu()
        B = model_output.shape[0]
        H_0 = (torch.inverse(model_output) @ sample)
        gamma0 = self.gamma0[timestep].to(device)
        gamma1 = self.gamma1[timestep].to(device)
        sample = se3.exp(gamma0 * se3.log(H_0) + gamma1 * se3.log(sample))
        return sample

    def denoise_equiv(self,
                model_output, # [B, 4, 4]
                timestep,  
                sample, # (B, pred_horizon, 4, 4)
                device):

        timestep = timestep[0].to(device)
        B = model_output.shape[0]

        H_0 = torch.inverse(model_output) @ sample.to(device)

        return H_0