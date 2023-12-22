import torch
import torchaudio
from torchaudio.functional import convolve, fftconvolve
import torch.nn.functional as F
from typing import Union


class ConvolutionReverb:
    def __init__(
        self,
        ir_file: str,
        conv_method: str = "fft",
        mix: float = 0.5
    ) -> None:
        self.ir_file: str = ir_file
        self.conv_method: str = conv_method
        self.mix = mix

        self.ir_sig: Union[None, torch.Tensor] = None
        self.ir_sr: Union[None, int] = None
        self.load_ir()

    def load_ir(self) -> None:
        self.ir_sig, _ = torchaudio.load(self.ir_file)
        non_zero_indices = torch.nonzero(self.ir_sig.squeeze())
        start_index = non_zero_indices[0]
        end_index = non_zero_indices[-1] + 1
        self.ir_sig = self.ir_sig[start_index:end_index]
        
    def convolve(self, input_sig: torch.Tensor) -> torch.Tensor:
        length = input_sig.shape[1] + self.ir_sig.shape[1] - 1

        if self.conv_method == "fft":
            wet_sig = fftconvolve(input_sig, self.ir_sig, mode="full")
        elif self.conv_method == "direct":
            wet_sig = convolve(input_sig, self.ir_sig, mode="full")
        else:
            raise ValueError("Invalid convolution method.")

        input_sig = F.pad(input_sig, (0, length - input_sig.shape[1]))
        output_sig = wet_sig * self.mix + input_sig * (1 - self.mix)
        output_sig = output_sig / output_sig.abs().max()
        return output_sig


if __name__ == "__main__":
    input_file: str = "../audio/raw/plk-fm-base.wav"
    output_file: str = "../audio/proc/output_convolve.wav"
    ir_file: str = "../audio/ir_analog/IR_AKG_BX25_1500ms_48kHz24b.wav"
    conv_method: str = "fft"
    wet_mix: float = 0.5

    # Load input signal
    input_sig, input_sr = torchaudio.load(input_file)

    reverb = ConvolutionReverb(ir_file, conv_method, wet_mix)
    output_sig = reverb.convolve(input_sig)

    torchaudio.save(
        uri=output_file,
        src=output_sig,
        sample_rate=input_sr,
        channels_first=True,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=24)

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure()
    plt.subplot(3, 1, 1)
    time_axis = torch.arange(0, input_sig.shape[1]) / input_sr
    time_axis = time_axis.unsqueeze(0)
    plt.plot(time_axis, input_sig.numpy(), label="input", alpha=0.5)
    plt.plot(time_axis, output_sig.numpy(), label="output", alpha=0.5)
    # plt.legend()
    plt.subplot(3, 1, 2)
    plt.specgram(input_sig.numpy(), Fs=input_sr, cmap='inferno')
    plt.subplot(3, 1, 3)
    plt.specgram(output_sig.numpy(), Fs=input_sr, cmap='inferno')
    plt.tight_layout()
    plt.show()
