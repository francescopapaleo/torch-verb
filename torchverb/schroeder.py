import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F


class SchroederReverb(nn.Module):
    def __init__(self, n_allpass, sample_rate, mix=0.5):
        super().__init__()
        self.n_allpass = n_allpass
        self.sample_rate = sample_rate
        self.mix = mix
        # Creating a list of allpass filter parameters
        self.allpass_params = self._create_allpass_params()

    def _create_allpass_params(self):
        # This function initializes parameters for each allpass filter
        params = []
        for _ in range(self.n_allpass):
            central_freq = (
                torch.rand(1).item() * 2000 + 500
            )  # Random frequency between 500 and 2500 Hz
            Q = torch.rand(1).item() * 0.5 + 0.5  # Random Q between 0.5 and 1.0
            params.append((central_freq, Q))
        return params

    def forward(self, input_sig):
        output_sig = input_sig
        for central_freq, Q in self.allpass_params:
            output_sig = F.allpass_biquad(output_sig, self.sample_rate, central_freq, Q)

        # Mix wet and dry signals
        mixed_sig = output_sig * self.mix + input_sig * (1 - self.mix)
        # Normalize if required
        mixed_sig = mixed_sig / mixed_sig.abs().max()

        return mixed_sig


if __name__ == "__main__":
    input_file: str = "./audio/raw/plk-fm-base.wav"
    output_file: str = "./audio/proc/output_schroeder.wav"

    input_sig, input_sr = torchaudio.load(input_file)

    # Create the reverb module
    reverb = SchroederReverb(n_allpass=400, sample_rate=input_sr, mix=0.5)
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
    plt.plot(input_sig[0, :].numpy(), label="Input", alpha=0.5)
    plt.plot(output_sig[0, :].numpy(), label="Output", alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()
