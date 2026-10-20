import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as Func

from torch import Tensor
from typing import Tuple, List, Union


class STFTEncoder(nn.Module):
    def __init__(
            self, 
            n_fft: int = 512, 
            hop_length: int = 256, 
            win_length: int = 512,
            return_complex: bool = False
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.return_complex = return_complex

        window = torch.hann_window(win_length).pow(0.5)
        self.register_buffer("window", window)

    def forward(self, wav: Tensor) -> Tensor:
        # wav: (B, T)
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, 
                          return_complex=self.return_complex)
        
        if self.return_complex:
            return spec  # (B, F, T) -> complex spectrogram
        else:
            return spec.permute(0, 3, 2, 1)  # (B, 2, T, F) -> real and imaginary parts


class STFTDecoder(nn.Module):
    def __init__(
            self, 
            n_fft: int = 512, 
            hop_length: int = 256, 
            win_length: int = 512
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(win_length).pow(0.5)
        self.register_buffer("window", window)

    def forward(self, spec: Tensor) -> Tensor:
        # spec: (B, F, T) or (B, F, T, 2)
        if len(spec.shape) == 3:
            wav = torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)

        elif len(spec.shape) == 4:
            assert spec.shape[-1] == 2, "The last dimension must be 2 for real and imaginary parts."
            spec_complex = torch.complex(spec[..., 0], spec[..., 1])
            wav = torch.istft(spec_complex, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)

        else:
            raise ValueError("Input spectrogram must be 3D or 4D tensor.")

        return wav  # (B, T)


class SFE(nn.Module):
    def __init__(
            self, 
            kernel_size: int,
            stride: int
    ):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=stride, padding=(0, (kernel_size - 1) // 2))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, F)
        x_unfold = self.unfold(x)  # (B, C*kernel_size, F * T)
        x_sfe = x_unfold.contiguous().view(x.shape[0], x_unfold.shape[1], x.shape[2], x.shape[3])
        return x_sfe  # (B, C*kernel_size, T, F)
    

class ERB(nn.Module):
    def __init__(
            self, 
            erb_subband_num: int, 
            nfft: int = 512, 
            fs: float = 16000.0, 
            low_lim: int = 2000, 
            high_lim: int = 8000
    ):
        super().__init__()
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = round(low_lim / fs * nfft)
        erb_subband_2 = erb_subband_num - self.erb_subband_1
        erb_filters = self.erb_filter_banks(
            self.erb_subband_1, erb_subband_2, nfft=nfft, fs=fs, low_lim=low_lim, high_lim=high_lim
        )
        self.erb_fc = nn.Linear(nfreqs, erb_subband_num, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_num, nfreqs, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

        self.act = nn.Sigmoid()

    def hz2erb(self, freq_hz):
        erb_f = 21.4 * np.log10(0.00437 * freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10 ** (erb_f / 21.4) - 1) / 0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, fs=16000, low_lim=2000, high_lim=8000):
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = \
            (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = \
                (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = \
                (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) / (bins[i + 2] - bins[i + 1] + 1e-12)
        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[-2, bins[-2]:bins[-1] + 1]

        return torch.from_numpy(np.abs(erb_filters[:, erb_subband_1:]))

    def erb(self, x):
        """x: (B,C,T,F)"""
        return torch.concat((x[..., :self.erb_subband_1],
                             self.erb_fc(x[..., self.erb_subband_1:])), dim=-1)

    def ierb(self, x_erb, flag=False):
        """x: (B,C,T,F_erb)"""
        y = torch.concat((x_erb[..., :self.erb_subband_1],
                          self.ierb_fc(x_erb[..., self.erb_subband_1:])), dim=-1)
        if flag:
            return self.act(y)
        else:
            return y


class TRA(nn.Module):
    def __init__(
            self, 
            channels: int,
            channel_mult: int = 2
    ):
        super(TRA, self).__init__()
        inter_channels = channels * channel_mult

        self.att_gru = nn.GRU(channels, inter_channels, 1, batch_first=True)
        self.att_fc = nn.Linear(inter_channels, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, F)
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class LayerNorm(nn.Module):
    def __init__(
            self, 
            c: int, 
            f: int
    ):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm([c, f], eps=1e-8)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, F]
        return self.ln(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)


class ChompT(nn.Module):
    def __init__(
            self, 
            t: int
    ):
        super(ChompT, self).__init__()
        self.t = t

    def forward(self, x: Tensor) -> Tensor:
        return x[..., :-self.t, :]


class AffinePReLU(nn.Module):
    def __init__(
            self, 
            channels: int, 
            width: int, 
            init: float = 0.25
    ):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(channels, width))
        self.affine_bias = nn.Parameter(torch.zeros(channels, width))
        self.slope_weight = nn.Parameter(torch.empty(channels))
        nn.init.constant_(self.slope_weight, init)
    
    def forward(self, x: Tensor) -> Tensor:
        y = self.affine_weight[None,:,None,:] * x + self.affine_bias[None,:,None,:]
        y = y + torch.where(x>0, x, self.slope_weight.view(1,-1,1,1) * x)
        return y
    

class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nfreq: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        use_deconv: bool = False,
        separable: bool = False,
        bn: bool = True,
        act: bool = True,
        act_type: str = "affine_prelu"
    ):
        layers = []
        tpad_size = (kernel_size[0] - 1) * dilation[0]
        Groups = out_channels if separable else groups
        In_channels = out_channels if separable else in_channels
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        if separable: layers.append(nn.Conv2d(in_channels, out_channels, (1, 1)))
        if not use_deconv: layers.append(torch.nn.ConstantPad2d([0, 0, tpad_size, 0], 0.0))
        layers.append(conv_module(
            In_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=Groups
        ))
        if use_deconv and tpad_size != 0: layers.append(ChompT(tpad_size))

        if bn: layers.append(nn.BatchNorm2d(out_channels))

        if act:
            assert act_type in ["prelu", "affine_prelu"]

            if act_type == "prelu":
                layers.append(nn.PReLU(out_channels))
            else:
                layers.append(AffinePReLU(out_channels, nfreq))
        
        super().__init__(*layers)


class MobileNetBlockv2(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_channels: int,
        nfreq: int,
        kernel_size: Tuple[int, int] = (3, 3),
        padding: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        use_deconv: bool = False,
        act_type: str = "prelu"
    ):
        super().__init__()
        freq_before = 2 * nfreq - 1 if not use_deconv else (nfreq + 1) // 2

        self.pwconv1 = ConvBlock(in_channels=in_ch, out_channels=hidden_channels,
                           nfreq=freq_before, kernel_size=(1, 1), bn=True, act=True, act_type=act_type)
        
        self.dwconv = ConvBlock(in_channels=hidden_channels, out_channels=hidden_channels,
                                nfreq=nfreq, kernel_size=kernel_size, padding=padding,
                                stride=stride, dilation=dilation, groups=hidden_channels,
                                bn=True, act=True, use_deconv=use_deconv, act_type=act_type)
        
        self.pwconv2 = ConvBlock(in_channels=hidden_channels, out_channels=out_ch,
                           nfreq=nfreq, kernel_size=(1, 1), bn=True, act=False, act_type=act_type)
        
        self.tra = TRA(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = self.pwconv2(x)
        x = self.tra(x)
        return x


class GRNN(nn.Module):
    def __init__(
            self, 
            input_size: int, 
            hidden_size: int, 
            num_layers: int = 1, 
            bidirectional: bool = False
    ):
        super().__init__()
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=True, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        y1, h1 = self.rnn1(x1)
        y2, h2 = self.rnn2(x2)
        return torch.cat((y1, y2), dim=-1), torch.cat((h1, h2), dim=0)
        

class InterRNN(nn.Module):
    def __init__(
            self, 
            channels: int, 
            nfreqs: int
    ):
        super().__init__()
        self.inter_rnn = nn.GRU(input_size=channels, hidden_size=channels, bidirectional=False, batch_first=True)
        self.inter_fc = nn.Linear(channels, channels)
        self.inter_ln = nn.LayerNorm(((nfreqs, channels)), eps=1e-8)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, F]
        bs, ch, nt, nf = x.shape
        ref = x
        inter_x = x.permute(0, 3, 2, 1)  # [B, F, T, C]
        inter_x = inter_x.reshape(bs * nf, nt, ch)  # [B*F, T, C]
        inter_x = self.inter_rnn(inter_x)[0]  # [B*F, T, C]
        inter_x = self.inter_fc(inter_x)  # [B*F, T, C]
        inter_x = inter_x.reshape(bs, nf, nt, ch)  # [B, F, T, C]
        inter_x = inter_x.permute(0, 2, 1, 3)  # [B, T, F, C]
        inter_x = self.inter_ln(inter_x)  # [B, T, F, C]
        inter_out = torch.add(ref, inter_x.permute(0, 3, 1, 2))  # [B, C, T, F]
        return inter_out


class CrossPathFusion(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            hidden_size: int,
            out_features: int
    ):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.ln = nn.LayerNorm([hidden_size], eps=1e-8)
        self.act = nn.ELU() 
        
        self.rnn = GRNN(input_size=hidden_size, hidden_size=hidden_size, bidirectional=False)

        self.fc_out = nn.Linear(hidden_size, out_features)

    def forward(
            self, 
            x1: Tensor, 
            x2: Tensor | None = None
    ) -> Tensor:
        # x1: (B, C, T, F)
        # x2: (B, C, T, F)
        x1_res, x2_res = x1, x2

        bs1, ch1, nt1, nf1 = x1.shape
        x1 = x1.permute(0, 2, 1, 3).flatten(2)  # (B, C, T, F) -> (B, T, C*F)

        if x2 is not None:
            bs2, ch2, nt2, nf2 = x2.shape
            x2 = x2.permute(0, 2, 1, 3).flatten(2)  # (B, C, T, F) -> (B, T, C*F)
            x = torch.cat((x1, x2), dim=-1)  # (B, T, H)

        else:
            x = x1
            
        x = self.fc(x)  # (B, T, H)
        x = self.ln(x)  
        x = self.act(x)
        x = self.rnn(x)[0]  # (B, T, H)
        x = self.fc_out(x)  # (B, T, F)

        if x2 is not None:
            x1, x2 = torch.split(x, [ch1*nf1, ch2*nf2], dim=-1)  # (B, T, F1), (B, T, F2)
            x1 = x1.reshape(bs1, nt1, ch1, nf1).permute(0, 2, 1, 3)  # (B, T, F1) -> (B, C, T, F)
            x2 = x2.reshape(bs2, nt2, ch2, nf2).permute(0, 2, 1, 3)  # (B, T, F2) -> (B, C, T, F)
            return x1 + x1_res, x2 + x2_res
        
        else:
            x = x.reshape(bs1, nt1, ch1, nf1).permute(0, 2, 1, 3)  # (B, T, F) -> (B, C, T, F)
            return x


class CoFiLite(nn.Module):
    def __init__(self):
        super().__init__()
        hc1 = 6
        hc2 = 6
        hh1 = 6
        hh2 = 6
        hf1 = 17
        hf2 = 33

        hidden_dim = 76

        act_type = "affine_prelu"

        self.stft_encoder = STFTEncoder(n_fft=512, hop_length=256, win_length=512)
        self.ERB = ERB(erb_subband_num=129, nfft=512, fs=16000, low_lim=2000, high_lim=8000)
        self.sfe = SFE(kernel_size=3, stride=1)

        self.en_erb1 = MobileNetBlockv2(3, hc1, hh1, 65, kernel_size=(3, 5), stride=(1, 2), padding=(0, 2), act_type=act_type)
        self.en_erb2 = MobileNetBlockv2(hc1, hc1, hh1, 33, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), act_type=act_type)
        self.en_erb3 = MobileNetBlockv2(hc1, hc1, hh1, 17, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), act_type=act_type)

        self.en_lfb1 = MobileNetBlockv2(9, hc2, hh2, 33, kernel_size=(3, 3), stride=(1, 2), padding=(0, 1), act_type=act_type)

        self.spgrnn = CrossPathFusion(hf1*hc1 + hf2*hc2, hidden_dim, hf1*hc1 + hf2*hc2)

        self.inter_rnn1 = InterRNN(hc1, hf1)
        self.inter_rnn2 = InterRNN(hc2, hf2)
        self.inter_rnn3 = InterRNN(hc1, hf1)
        self.inter_rnn4 = InterRNN(hc2, hf2)
    
        self.de_erb3 = MobileNetBlockv2(hc1, hc1, hh1, 33, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, act_type=act_type)
        self.de_erb2 = MobileNetBlockv2(hc1, hc1, hh1, 65, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, act_type=act_type)
        self.de_erb1 = MobileNetBlockv2(hc1, 1, hh1, 129, kernel_size=(3, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, act_type=act_type)
        
        self.de_lfb1 = MobileNetBlockv2(hc2, 1, hh2, 65, kernel_size=(3, 3), stride=(1, 2), padding=(0, 1), use_deconv=True, act_type=act_type)

        self.out_act = nn.Sigmoid()
        self.stft_decoder = STFTDecoder(n_fft=512, hop_length=256, win_length=512)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, T]  time domain representation
        """
        n_samples = x.shape[-1]
        x = self.stft_encoder(x)  # [B, 2, T, F]
        x_ref = x.permute(0, 3, 2, 1)  # [B, F, T, 2]

        X_amp = torch.sqrt(x[:, 0].pow(2) + x[:, 1].pow(2) + 1e-12).unsqueeze(1)  # [B, 1, T, F = 257]
        X_ric = x.div(X_amp.pow(0.7))  # [B, 2, T, F = 257]
        X_log = torch.log10(X_amp)  # [B, 1, T, F = 257]
        x = torch.cat([X_log, X_ric], dim=1)  # [B, 3, T, F = 257]

        x_erb = self.ERB.erb(X_amp)  # [B, 1, T, F = 129]
        x_erb = torch.log10(x_erb)  # [B, 1, T, F = 129]
        x_erb = self.sfe(x_erb)  # [B, 3, T, F = 129]
        
        en_x1 = self.en_erb1(x_erb)  # [B, C, T, F = 65]
        en_x2 = self.en_erb2(en_x1)  # [B, C, T, F = 33]
        en_x3 = self.en_erb3(en_x2)  # [B, C, T, F = 17]
        emb_erb = self.inter_rnn1(en_x3)  # [B, C, T, 17]
        
        x_f = x[..., :65]  # [B, 3, T, 65]
        x_f = self.sfe(x_f)  # [B, 9, T, 65]
        
        enf_x1 = self.en_lfb1(x_f)  # [B, C, T, F = 33]
        emb_lfb = self.inter_rnn2(enf_x1)  # [B, C, T, F = 33]

        emb1, emb2 = self.spgrnn(emb_erb, emb_lfb) 

        emb_erb = self.inter_rnn3(emb1)  # [B, C, T, F = 17]
        de_x3 = self.de_erb3(emb_erb + en_x3)  # [B, C, T, F = 33]
        de_x2 = self.de_erb2(de_x3 + en_x2)  # [B, C, T, F = 65]
        de_x1 = self.de_erb1(de_x2 + en_x1)  # [B, C = 1, T, F = 129]
        de_x1 = self.out_act(de_x1)  # [B, C = 1, T, F = 129]

        emb_lfb = self.inter_rnn4(emb2)  # [B, C, T, F = 33]
        def_x1 = self.de_lfb1(emb_lfb + enf_x1)  # [B, C = 1, T, F = 65]
        def_x1 = self.out_act(def_x1)  # [B, C = 1, T, F = 65]

        est_erb = self.ERB.ierb(de_x1).permute(0, 3, 2, 1).contiguous()  # [B, C, T, F] -> [B, F, T, C]
        enh0 = x_ref * est_erb  # envelope enhancement
        mask = def_x1.permute(0, 3, 2, 1)
        enh = torch.cat([enh0[:, :65] * mask, enh0[:, 65:]], dim=1)  # low-frequency bands enhancement
        enh_wav = self.stft_decoder(enh)
        enh_wav = Func.pad(enh_wav, (0, n_samples - enh_wav.shape[-1]))

        return enh_wav


if __name__ == "__main__":
    T = 16000
    inputs = torch.randn(1, T)
    model = CoFiLite()
    output = model(inputs)
    print(output.shape)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (T,), as_strings=False,
                                              print_per_layer_stat=True, verbose=False)
    
    print(f"The complexity of CoFi-Lite: MACs={macs/1e6:.2f}, Params={params/1e3:.2f} k\n")
    
    """
    causality check
    """
    a = torch.randn(1, T)
    b = torch.randn(1, T)
    c = torch.randn(1, T)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)
    y1 = model(x1)
    y2 = model(x2)
    print((y1[:, :(T - 256 * 2)] - y2[:, :(T - 256 * 2)]).abs().max())
    print((y1[:, T:] - y2[:, T:]).abs().max())
