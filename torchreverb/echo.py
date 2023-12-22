import torch
import torchaudio
import soundfile as sf
import numpy as np


class Echo:
    def __init__(
        self,
        sr: int,
        delay: float = 0.5,
        mix: float = 1.0
    ) -> None:
        self.delay_samples: int = round(delay * sr)
        self.mix: float = mix / 100

    def delay(self, input_sig: torch.Tensor) -> torch.Tensor:
        delay_array = torch.zeros(self.delay_samples, dtype=input_sig.dtype)
        wet_sig = torch.cat((delay_array, input_sig))
        dry_sig = torch.cat((input_sig, delay_array))
        output_sig = wet_sig * self.mix + dry_sig * (1 - self.mix)
        return output_sig


if __name__ == "__main__":
    input_file: str = "../audio/plk-fm-base.wav"
    output_file: str = "../output_echo.wav"
    wet_mix: float = 0.5

    # Load input signal
    input_sig, input_sr = torchaudio.load(input_file)

    # Create Echo instance
    echo = Echo(input_sr)

    # Apply delay to the input signal
    output_sig = echo.delay(input_sig)

    # Save the resulting audio with explicit dtype and format
    sf.write(output_file, output_sig.numpy(), input_sr, format='WAV', subtype='PCM_24')

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(input_sig.numpy(), label="input", alpha=0.5)
    plt.plot(output_sig.numpy(), label="output", alpha=0.5)
    # plt.legend()
    plt.subplot(3, 1, 2)
    plt.specgram(input_sig.numpy(), Fs=input_sr, cmap='inferno')
    plt.subplot(3, 1, 3)
    plt.specgram(output_sig.numpy(), Fs=input_sr, cmap='inferno')
    plt.tight_layout()
    plt.show()
