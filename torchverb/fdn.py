""" 
Copyright (C) 2024 Francesco Papaleo
Distributed under the GNU Affero General Public License v3.0
"""
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F


class FDNReverb(nn.Module):
    """
    FDNReverb class represents a Feedback Delay Network (FDN) reverb module.

    Parameters
    ----------
    delays : list
        A list of delay times in seconds.
    sample_rate : int
        The sample rate of the audio signal.
    mix : float, optional
        The mix ratio between the wet and dry signals. Defaults to 0.5.

    Attributes
    ----------
    sample_rate : int
        The sample rate of the audio signal.
    mix : float
        The mix ratio between the wet and dry signals.
    delays : list
        A list of delay times in samples.
    feedback_gain : torch.nn.Parameter
        The feedback gains for each delay line.
    orthogonal_matrix : torch.nn.Parameter
        The orthogonal mixing matrix.
    delay_buffers : list
        The delay buffers for each delay line.
    """
    def __init__(self, delays, sample_rate, mix=0.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.mix = mix

        # Convert delay times to samples and ensure they are integers
        self.delays = [int(delay * sample_rate) for delay in delays]

        # Initialize feedback gains
        self.feedback_gain = nn.Parameter(torch.rand(len(self.delays)))

        # Initialize orthogonal matrix using PyTorch's QR decomposition
        random_matrix = torch.randn(len(self.delays), len(self.delays))
        q, _ = torch.linalg.qr(random_matrix)
        self.orthogonal_matrix = nn.Parameter(q.float())

        # Initialize delay buffers
        max_delay = max(self.delays)
        self.delay_buffers = [torch.zeros(max_delay) for _ in range(len(self.delays))]

    def forward(self, input_sig):
        """
        Apply the FDN reverb effect to the input signal.

        Parameters
        ----------
        input_sig : torch.Tensor
            The input audio signal.

        Returns
        -------
        torch.Tensor
            The output audio signal with the FDN reverb effect applied.

        """
        num_samples = input_sig.size(-1)
        output_sig = torch.zeros_like(input_sig)

        # Processing each delay line
        processed_signals = torch.zeros((len(self.delays), num_samples))

        for i, delay in enumerate(self.delays):
            delay_buffer = self.delay_buffers[i]
            delayed_signal = torch.cat([delay_buffer[-delay:], input_sig[0, :-delay]])

            # Update delay buffer
            self.delay_buffers[i] = torch.cat(
                [delay_buffer[num_samples:], input_sig[0, :]], dim=0
            )

            # Feedback and mixing
            processed_signals[i, :] = delayed_signal * self.feedback_gain[i]

        # Apply orthogonal mixing matrix
        processed_signals = torch.matmul(self.orthogonal_matrix, processed_signals)

        # Summing the processed signals
        output_sig[0, :] = torch.sum(processed_signals, dim=0)

        # Mix wet and dry signals
        output_sig = output_sig * self.mix + input_sig * (1 - self.mix)
        output_sig = output_sig / torch.max(torch.abs(output_sig))

        return output_sig


if __name__ == "__main__":
    input_file: str = "./audio/raw/plk-fm-base.wav"
    output_file: str = "./audio/proc/output_fdn.wav"

    # Load the audio file
    input_sig, input_sr = torchaudio.load(input_file)

    delays = [0.1, 0.2, 0.3, 0.4]

    # Create an instance of the FeedbackDelayNetwork
    fdn = FDNReverb(delays, sample_rate=input_sr)
    output_sig = fdn(input_sig)

    torchaudio.save(output_file, output_sig, input_sr, channels_first=True)

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure()
    plt.plot(input_sig[0, :].numpy(), label="Input", alpha=0.5)
    plt.plot(output_sig[0, :].detach().numpy(), label="Output", alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()
