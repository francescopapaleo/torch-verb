import torch
import torchaudio
import torchaudio.functional as F
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
        self.mix: float = mix / 100

        self.ir_sig: Union[None, torch.Tensor] = None
        self.ir_sr: Union[None, int] = None
        self.load_ir()

    def load_ir(self) -> None:
        ir_sig, ir_sr = torchaudio.load(self.ir_file)
        self.ir_sig = ir_sig
        self.ir_sr = ir_sr

    def convolve(self, input_sig: torch.Tensor) -> torch.Tensor:
        input_sig = input_sig / torch.linalg.vector_norm(input_sig, ord=2)
        self.ir_sig = self.ir_sig / torch.linalg.vector_norm(self.ir_sig, ord=2)
        if self.conv_method == "fft":
            wet_sig = F.fftconvolve(input_sig, self.ir_sig, mode="same")
        elif self.conv_method == "direct":
            wet_sig = F.convolve(input_sig, self.ir_sig, mode="same")
        else:
            raise ValueError("Invalid convolution method.")
        output_sig = wet_sig * self.mix + input_sig * (1 - self.mix)
        print(output_sig.shape)
        return output_sig


if __name__ == "__main__":
    input_file: str = "../audio/raw/plk-fm-base.wav"
    output_file: str = "../audio/proc/output_convolve.wav"
    ir_file: str = "../audio/ir_analog/IR_AKG_BX25_1500ms_48kHz24b.wav"
    conv_method: str = "fft"
    wet_mix: float = 1.0

    # Load input signal
    input_sig, input_sr = torchaudio.load(input_file)

    # Create ConvolutionReverb instance
    reverb = ConvolutionReverb(ir_file, conv_method, wet_mix)

    # Convolve input signal
    output_sig = reverb.convolve(input_sig)

    # Save the resulting audio with explicit dtype and format
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
    plt.plot(time_axis, input_sig.numpy(), label="input", alpha=0.5)
    plt.plot(time_axis, output_sig.numpy(), label="output", alpha=0.5)
    # plt.legend()
    plt.subplot(3, 1, 2)
    plt.specgram(input_sig.numpy(), Fs=input_sr, cmap='inferno')
    plt.subplot(3, 1, 3)
    plt.specgram(output_sig.numpy(), Fs=input_sr, cmap='inferno')
    plt.tight_layout()
    plt.show()
