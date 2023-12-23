import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt


def damping_filter_coeffs(delays, t_60, alpha):
    element_1 = torch.log10(torch.tensor(10.0) / 4)
    element_2 = 1 - (1 / (alpha ** 2))
    g = 10 ** ((-3 * delays * (1/48000)) / t_60)
    p = element_1 * element_2 * torch.log10(g)
    print(g)
    print(p)
    return p, g


def delay(input_signal, delay, gain=1):
    output_signal = F.pad(input_signal, (delay, 0), value=0)
    output_signal = output_signal * gain
    return output_signal[:input_signal.size(0)]


def damping_filter(input_signal, p, g):
    B = g * (1 - p)
    A = torch.tensor([1, -p], dtype=torch.float64)
    output_signal = F.lfilter(B, A, input_signal)
    return output_signal


def tonal_correction_filter(input_signal, alpha):
    beta = (1 - alpha) / (1 + alpha)
    E_nomin = torch.tensor([1, -beta], dtype=torch.float64)
    E_denomin = torch.tensor([1 - beta], dtype=torch.float64)
    output_signal = F.lfilter(E_nomin, E_denomin, input_signal)
    return output_signal


class FilterDelayNetwork(torch.nn.Module):
    def __init__(self, delays, t_60, alpha):
        super().__init__()
        self.delays = delays
        self.t_60 = t_60
        self.alpha = alpha
        self.p, self.g = damping_filter_coeffs(self.delays, self.t_60, self.alpha)

    def forward(self, input_signal):
        output_signal = input_signal
        for delay in self.delays:
            output_signal = delay(output_signal, delay)
        output_signal = damping_filter(output_signal, self.p, self.g)
        output_signal = tonal_correction_filter(output_signal, self.alpha)
        return output_signal


if __name__ == "__main__":
    input_file: str = "./audio/raw/plk-fm-base.wav"
    output_file: str = "./audio/proc/output_fdn.wav"

    # Load the audio file
    waveform, sample_rate = torchaudio.load(input_file)

    # Define the parameters for the FilterDelayNetwork
    delays = [100, 200, 300, 400]  # Replace with your values
    t_60 = 0.6  # Replace with your value
    alpha = 0.2  # Replace with your value

    # Create an instance of the FilterDelayNetwork
    fdn = FilterDelayNetwork(delays, t_60, alpha)

    # Apply the FilterDelayNetwork to the audio data
    output_signal = fdn(waveform)

    # Save the output signal to a file
    torchaudio.save(output_file, output_signal, sample_rate)
