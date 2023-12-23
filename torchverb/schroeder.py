import torch
import torch.nn as nn
import torchaudio
from typing import List


class SchroederReverb(nn.Module):
    def __init__(
        self,
        allpass_delays: List[int],
        comb_delays: List[int],
        comb_feedbacks: List[float],
    ) -> None:
        super().__init__()
        self.allpass_delays = allpass_delays
        self.comb_delays = comb_delays
        self.comb_feedbacks = comb_feedbacks

        # Input validation
        if not all(
            len(lst) == len(allpass_delays)
            for lst in [comb_delays, comb_feedbacks]
        ):
            raise ValueError("Mismatched lengths of delay and feedback lists.")

    def allpass_filter(self, x, N, g):
        y = torch.zeros_like(x)
        for n in range(N, len(x)):
            y[n] = -g * y[n - N] + x[n] + g * x[n - N]
        return y

    def feedback_comb_filter(self, x, N, g):
        y = torch.zeros_like(x)
        for n in range(N, len(x)):
            y[n] = x[n] + g * y[n - N]
        return y

    def mixing_matrix(self, x):
        return torch.eye(x.size(0), dtype=x.dtype, device=x.device)

    def forward(self, x):
        # Apply allpass filters in series
        for delay in self.allpass_delays:
            x = self.allpass_filter(x, delay, 0.7)

        # Apply parallel bank of feedback comb filters
        comb_output = torch.zeros_like(x)
        for delay, feedback in zip(self.comb_delays, self.comb_feedbacks):
            comb_output += self.feedback_comb_filter(x, delay, feedback)

        # Ensure output tensor has the correct shape [n_channels, time]
        comb_output = comb_output[:, :x.shape[1]]

        # Apply mixing matrix
        mixed_output = self.mixing_matrix(x) @ comb_output

        return mixed_output


if __name__ == "__main__":
    input_file: str = "../audio/raw/plk-fm-base.wav"
    output_file: str = "../audio/proc/output_schroeder.wav"

    # Load input signal
    input_sig, input_sr = torchaudio.load(input_file)

    # Set parameters
    allpass_delays = [50, 120, 200]
    comb_delays = [137, 277, 422]
    comb_feedbacks = [0.84, 0.88, 0.92]

    reverb = SchroederReverb(allpass_delays, comb_delays, comb_feedbacks)
    output_sig = reverb(input_sig)

    # Save the resulting audio with explicit dtype and format
    torchaudio.save(
        uri=output_file,
        src=output_sig,
        sample_rate=input_sr,
        channels_first=True,
        format="wav",
        encoding="PCM_S",
        bits_per_sample=24)
