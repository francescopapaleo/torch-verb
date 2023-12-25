import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F


class SchroederReverb(nn.Module):
    def __init__(self, sample_rate, mix) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mix = mix

    def forward(self, input_sig: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    input_file: str = "../audio/raw/plk-fm-base.wav"
    output_file: str = "../audio/proc/output_convolve.wav"

    input_sig, input_sr = torchaudio.load(input_file)

    # Create the reverb module
    reverb = SchroederReverb(input_sr, 0.5)
    output_sig = reverb(input_sig)

    torchaudio.save(
        uri=output_file,
        src=output_sig,
        sample_rate=input_sr,
        channels_first=True,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=24,
    )

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure()

    plt.legend()
    plt.grid(True)
    plt.show()
