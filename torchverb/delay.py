""" 
Copyright (C) 2024 Francesco Papaleo
Distributed under the GNU Affero General Public License v3.0
"""
import torch
import torch.nn as nn
import torchaudio


class DelayLine(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        delays: list,
        mix: float,
    ) -> None:
        """
        DelayLine class represents a delay line module.

       Parameters
        ----------
        sample_rate : int
            The sample rate of the audio signal.
        delays : list
            A list of delay times in seconds.
        mix : float
            The mix ratio between the wet and dry signals.

        """
        super().__init__()
        self.sample_rate = sample_rate
        self.delay_list: list = [round(delay * sample_rate) for delay in delays]
        self.mix = mix

    def delay_samples(self):
        return max(self.delay_list)

    def delay(self, input_sig: torch.Tensor, delay_samples: int) -> torch.Tensor:
        """
        Apply delay to the input signal.

        Parameters
        ----------
        input_sig : torch.Tensor
            The input audio signal.
        delay_samples : int
            The number of samples to delay the signal.

        Returns
        -------
        torch.Tensor
            The delayed audio signal.

        """
        delay_array = torch.zeros([1, delay_samples])

        dry_sig = torch.cat([input_sig, delay_array], dim=-1)
        wet_sig = torch.cat([delay_array, input_sig], dim=-1)

        output_sig = wet_sig * self.mix + dry_sig * (1 - self.mix)
        output_sig = output_sig / output_sig.abs().max()

        output_sig = output_sig[:, : input_sig.size(1)]
        return output_sig

    def forward(self, input_sig: torch.Tensor) -> torch.Tensor:
        """
        Apply the delay line effect to the input signal.

        Parameters
        ----------
        input_sig : torch.Tensor
            The input audio signal.

        Returns
        -------
        torch.Tensor
            The output audio signal with the delay effect applied.

        """
        delayed_sigs = []
        for delay in self.delay_list:
            delayed_sigs.append(self.delay(input_sig, delay))
        sum_sigs = torch.stack(delayed_sigs).sum(dim=0)
        sum_sigs = sum_sigs / sum_sigs.abs().max()
        return sum_sigs


if __name__ == "__main__":
    input_file: str = "./audio/raw/plk-fm-base.wav"
    output_file: str = "./audio/proc/output_delay.wav"
    delays: list = [0.1, 0.2, 0.3, 0.4]
    mix: float = 0.5

    input_sig, input_sr = torchaudio.load(input_file)

    delay = DelayLine(sample_rate=input_sr, delays=delays, mix=mix)
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
