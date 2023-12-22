import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


def allpass_filter(x, N, g):
    y = torch.zeros_like(x)
    for n in range(N, len(x)):
        y[n] = -g * y[n - N] + x[n] + g * x[n - N]
    return y


def feedback_comb_filter(x, N, g):
    y = torch.zeros_like(x)
    for n in range(N, len(x)):
        y[n] = x[n] + g * y[n - N]
    return y


def mixing_matrix(x):
    return torch.eye(x.size(0), dtype=x.dtype, device=x.device)


def schroeder_reverberator(
    in_signal,
    allpass_delays,
    comb_delays,
    comb_feedbacks
):
    # Apply allpass filters in series
    allpass_output = in_signal
    for delay in allpass_delays:
        allpass_output = allpass_filter(allpass_output, delay, 0.7)

    # Apply parallel bank of feedback comb filters
    comb_outputs = [feedback_comb_filter(allpass_output, delay, feedback) for delay, feedback in zip(comb_delays, comb_feedbacks)]
    
    # Ensure output tensor has the correct shape (num_channels, num_samples)
    output = torch.stack(comb_outputs)

    # Apply mixing matrix
    output = mixing_matrix(output)

    return output


if __name__ == "__main__":
    input_file = "your_input_audio_file.wav"
    output_file = "output_reverberated_audio_file.wav"

    # Load input signal
    input_signal, sr = torchaudio.load(input_file)

    # Set parameters
    allpass_delays = [50, 120, 200]
    comb_delays = [137, 277, 422, 623]
    comb_feedbacks = [0.84, 0.88, 0.92, 0.96]

    # Apply Schroeder Reverberator
    output_signal = schroeder_reverberator(input_signal, allpass_delays, comb_delays, comb_feedbacks)

    # Save the resulting audio with explicit dtype and format
    torchaudio.save(output_file, output_signal, sr, format='wav', subtype='PCM_24')
