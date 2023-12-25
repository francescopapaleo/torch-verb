import torch
import torch.nn as nn
import torchaudio


class DelayLine(nn.Module):
    def __init__(
        self,
        sr: int,
        delay: float,
        mix: float,
    ) -> None:
        super().__init__()
        self.delay_samples: int = round(delay * sr)
        self.mix = mix

    def forward(self, input_sig: torch.Tensor) -> torch.Tensor:
        # tensor.shape: [n_channels, time]
        delay_array = torch.zeros([1, self.delay_samples])

        dry_sig = torch.cat([input_sig, delay_array], dim=-1)
        wet_sig = torch.cat([delay_array, input_sig], dim=-1)

        output_sig = wet_sig * self.mix + dry_sig * (1 - self.mix)
        output_sig = output_sig / output_sig.abs().max()
        return output_sig


if __name__ == "__main__":
    input_file: str = "../audio/raw/plk-fm-base.wav"
    output_file: str = "../audio/proc/output_echo.wav"
    delay: float = 0.25
    mix: float = 0.5

    input_sig, input_sr = torchaudio.load(input_file)

    delay = DelayLine(sr=input_sr, delay=delay, mix=mix)
    output_sig = delay(input_sig)

    torchaudio.save(
        uri=output_file,
        src=output_sig,
        sample_rate=input_sr,
        channels_first=True,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=24,
    )

    # Pad the shorter signal to match the length of the longer one
    max_length = max(input_sig.size(1), output_sig.size(1))
    input_sig = nn.functional.pad(input_sig, (0, max_length - input_sig.size(1)))
    output_sig = nn.functional.pad(output_sig, (0, max_length - output_sig.size(1)))

    time = torch.linspace(0, max_length / input_sr, max_length)

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure()
    # Plot the input_sig and output_sig waveforms overlapped on the same plot
    plt.plot(time.numpy(), input_sig[0, :].numpy(), label="Input signal", alpha=0.5)
    plt.plot(time.numpy(), output_sig[0, :].numpy(), label="Output signal", alpha=0.5)
    plt.legend()
    plt.show()
