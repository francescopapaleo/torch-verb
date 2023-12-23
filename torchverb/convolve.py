import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
from typing import Union


class ConvolutionReverb(nn.Module):
    def __init__(
        self,
        ir_file: str,
        conv_method: str = "fft",
        mix: float = 0.5
    ) -> None:
        super().__init__()
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

    def forward(self, input_sig: torch.Tensor) -> torch.Tensor:
        length = input_sig.shape[1] + self.ir_sig.shape[1] - 1

        if self.conv_method == "fft":
            wet_sig = F.fftconvolve(input_sig, self.ir_sig, mode="full")
        elif self.conv_method == "direct":
            wet_sig = F.convolve(input_sig, self.ir_sig, mode="full")
        else:
            raise ValueError("Invalid convolution method.")

        # Truncate or pad the wet signal to match the input signal length
        if wet_sig.shape[1] > input_sig.shape[1]:
            wet_sig = wet_sig[:, :input_sig.shape[1]]
        elif wet_sig.shape[1] < input_sig.shape[1]:
            padding = input_sig.shape[1] - wet_sig.shape[1]
            wet_sig = nn.functional.pad(wet_sig, (0, padding))

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
    output_sig = reverb(input_sig)

    torchaudio.save(
        uri=output_file,
        src=output_sig,
        sample_rate=input_sr,
        channels_first=True,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=24)

    # Pad the shorter signal to match the length of the longer one
    max_length = max(input_sig.size(1), output_sig.size(1))
    input_sig = nn.functional.pad(
        input_sig, (0, max_length - input_sig.size(1)))
    output_sig = nn.functional.pad(
        output_sig, (0, max_length - output_sig.size(1)))

    time = torch.linspace(0, max_length / input_sr, max_length)

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure()
    # Plot the input_sig and output_sig waveforms overlapped on the same plot
    plt.plot(
        time.numpy(), input_sig[0, :].numpy(),
        label='Input signal', alpha=0.5)
    plt.plot(
        time.numpy(), output_sig[0, :].numpy(),
        label='Output signal', alpha=0.5)
    plt.legend()
    plt.show()
