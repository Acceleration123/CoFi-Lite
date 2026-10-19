import torch
import torch.nn as nn
import typing
from typing import List
import functools


class HybridLoss(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        compress_factor=0.3,
        eps=1e-12,
        lamda_ri=30,
        lamda_mag=70):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.window = torch.hann_window(win_len)
        self.c = compress_factor
        self.eps = eps
        self.lamda_ri = lamda_ri
        self.lamda_mag = lamda_mag

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        
        device = y_true.device
        
        pred_stft = torch.stft(y_pred, self.n_fft, self.hop_len, self.win_len, self.window.to(device), return_complex=True)
        true_stft = torch.stft(y_true, self.n_fft, self.hop_len, self.win_len, self.window.to(device), return_complex=True)

        pred_mag = torch.abs(pred_stft).clamp(self.eps)
        true_mag = torch.abs(true_stft).clamp(self.eps)
        
        pred_stft_c = pred_stft / pred_mag**(1 - self.c)
        true_stft_c = true_stft / true_mag**(1 - self.c)

        real_loss = torch.mean((pred_stft_c.real - true_stft_c.real)**2)
        imag_loss = torch.mean((pred_stft_c.imag - true_stft_c.imag)**2)
        mag_loss = torch.mean((pred_mag**self.c - true_mag**self.c)**2)

        # SISNR loss
        y_norm = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)
        sisnr = - 2*torch.log10(
            torch.norm(y_norm, dim=-1, keepdim=True) / 
            torch.norm(y_pred - y_norm, dim=-1, keepdim=True).clamp(self.eps) + 
            self.eps
        ).mean()
        
        return self.lamda_ri*(real_loss + imag_loss) + self.lamda_mag*mag_loss + sisnr
    

if __name__=='__main__':
    a = torch.randn(2, 10000)
    b = torch.randn(2, 10000)

    loss_func = HybridLoss(n_fft=512, hop_len=256, win_len=512)
    loss = loss_func(a, b)
    print(loss)