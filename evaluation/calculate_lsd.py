import torch
import torch.nn as nn
import soundfile as sf

from torch import Tensor


def reader(audio_path: str) -> Tensor:
    audio, _ = sf.read(audio_path, dtype="float32")

    return torch.from_numpy(audio)



class LogSpectralDistance(nn.Module):
    def __init__(
            self,
            sr: float = 16000.0,
            n_fft: int = 512,
            hop_length: int = 256,
            win_length: int = 512,
            f_start: float = 0,
            f_end: float = 2000.0
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.window = torch.hann_window(window_length=win_length).pow(0.5)

        self.delta_f = sr / n_fft
        self.num_fbins = n_fft // 2 + 1

        self.start_index = int(f_start // self.delta_f)
        self.end_index = int(f_end // self.delta_f)

    def stft(
            self,
            audio: Tensor
    ) -> Tensor:
        device = audio.device  # (T,)
        return torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window.to(device), return_complex=False)  # (F, T)
    
    @staticmethod
    def logspec(spec: Tensor) -> Tensor:
        mag = torch.abs(spec)
        return torch.log10(mag.clamp(min=1e-8))
    
    def cal_lsd(
            self, 
            inf_logspec: Tensor,
            ref_logspec: Tensor
    ) -> Tensor:
        nt = inf_logspec.shape[-1]

        inf_usage = inf_logspec[self.start_index:self.end_index, :]
        ref_usage = ref_logspec[self.start_index:self.end_index, :]

        loss_tot = 0.0

        for i in range(nt):
            loss = torch.sum((inf_usage[:, i] - ref_usage[:, i]) ** 2)

            loss = loss / (self.end_index - self.start_index + 1)

            loss = torch.sqrt(loss)

            loss_tot += loss

        return loss_tot / nt

    def forward(
            self,
            inf: Tensor,
            ref: Tensor
    ) -> Tensor:
        assert len(inf) == len(ref)

        inf_spec = self.stft(inf)  # (F, T)
        ref_spec = self.stft(ref)  # (F, T)

        inf_logspec = self.logspec(inf_spec)
        ref_logspec = self.logspec(ref_spec)

        loss = self.cal_lsd(inf_logspec, ref_logspec)

        return loss.numpy()


if __name__ == "__main__":
    import librosa

    from tqdm import tqdm

    calculator = LogSpectralDistance()

    inf_dir = ""
    ref_dir = ""

    inf_paths = sorted(librosa.util.find_files(inf_dir, ext=["wav"]))
    ref_paths = sorted(librosa.util.find_files(ref_dir, ext=["wav"]))

    assert len(inf_paths) == len(ref_paths)

    loss_tot = 0.0

    for i in tqdm(range(inf_paths)):
        inf = reader(inf_paths[i])
        ref = reader(ref_paths[i])

        loss = calculator(inf, ref)

        loss_tot += loss

    loss_avg = loss_tot / len(inf_paths)





