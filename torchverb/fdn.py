import torch
import torchaudio
import torchaudio.functional as F
import torch.nn as nn
from delay import DelayLine


class FeedbackDelayNetwork(nn.Module):
    def __init__(self, delays, sample_rate=48000):
        super(FeedbackDelayNetwork, self).__init__()
        self.delays = delays
        self.sample_rate = sample_rate

        # Create the delay lines
        self.delay_lines = nn.ModuleList()
        for delay in delays:
            self.delay_lines.append(DelayLine(sample_rate, delay, 0.5))

    def forward(self, input_sig):
        delayed_signals = []
        for delay_line in self.delay_lines:
            delayed_signals.append(delay_line(input_sig))

        # Find the maximum size among delayed signals
        max_size = max([signal.size(0) for signal in delayed_signals])

        # Pad the signals to the maximum size before stacking
        delayed_signals_padded = [
            nn.functional.pad(signal, (0, max_size - signal.size(0)), value=0)
            for signal in delayed_signals
        ]

        # Stack and sum the delayed signals
        output_signal = torch.sum(torch.stack(delayed_signals_padded, dim=0), dim=0)

        return output_signal


if __name__ == "__main__":
    input_file: str = "./audio/raw/plk-fm-base.wav"
    output_file: str = "./audio/proc/output_fdn.wav"

    # Load the audio file
    waveform, sample_rate = torchaudio.load(input_file)
    print(waveform.shape)
    # Define the parameters for the FeedbackDelayNetwork
    delays = [100, 200, 300, 400]  # Replace with your values
    t_60 = 0.6  # Replace with your value
    alpha = 0.2  # Replace with your value

    # Create an instance of the FeedbackDelayNetwork
    fdn = FeedbackDelayNetwork(delays, t_60, alpha, sample_rate)

    # Apply the FeedbackDelayNetwork to the audio data
    output_signal = fdn(waveform)  # Assuming single-channel audio

    # Save the output signal to a file
    torchaudio.save(output_file, output_signal.unsqueeze(0), sample_rate)
