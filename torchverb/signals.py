""" 
Copyright (C) 2024 Francesco Papaleo
Distributed under the GNU Affero General Public License v3.0
"""
import torch
import torchaudio
import math
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def impulse(
    sample_rate: int,
    duration: float,
    decibels: float = -0.5,
    impulse_duration: float = 0.03,
) -> torch.Tensor:
    """
    Generates an impulse signal.

    Parameters
    ----------
    sample_rate : int
        The sample rate of the signal.
    duration : float
        The duration of the signal in seconds.
    decibels : float, optional
        The amplitude of the impulse in decibels. Defaults to -0.5.
    impulse_duration : float, optional
        The duration of the impulse in seconds. Defaults to 0.03.

    Returns
    -------
    torch.Tensor
        The generated impulse signal.
    """
    array_length = int(duration * sample_rate)
    impulse = torch.zeros(array_length)

    amplitude = 10 ** (decibels / 20)
    impulse_length = int(impulse_duration * sample_rate)

    start_idx = (array_length - impulse_length) // 2
    end_idx = start_idx + impulse_length

    impulse[start_idx:end_idx] = amplitude

    return impulse


def sine(
    sample_rate: int, duration: float, amplitude: float, frequency: float = 440.0
) -> torch.Tensor:
    """
    Generate a sine wave signal.

    Parameters
    ----------
    sample_rate : int
        The sample rate of the signal.
    duration : float
        The duration of the signal in seconds.
    amplitude : float
        The amplitude of the signal.
    frequency : float, optional
        The frequency of the sine wave in Hz. Defaults to 440.0.

    Returns
    -------
    torch.Tensor
        The generated sine wave signal.
    """
    t = torch.arange(0, duration, 1.0 / sample_rate)
    sine_wave = amplitude * torch.sin(2.0 * math.pi * frequency * t)
    return sine_wave


def log_sweep_tone(
    sample_rate: int,
    duration: float,
    amplitude: float,
    f0: float = 20,
    f1: float = 20000,
    inverse: bool = False,
) -> torch.Tensor:
    """
    Generate a logarithmic sweep tone signal.

    Parameters
    ----------
    sample_rate : int
        The sample rate of the signal.
    duration : float
        The duration of the signal in seconds.
    amplitude : float
        The amplitude of the signal.
    f0 : float, optional
        The starting frequency of the sweep tone. Defaults to 20.
    f1 : float, optional
        The ending frequency of the sweep tone. Defaults to 20000.
    inverse : bool, optional
        Whether to generate an inverse sweep tone. Defaults to False.

    Returns
    -------
    torch.Tensor
        The generated sweep tone signal.
    """
    R = torch.log(torch.tensor(f1 / f0))
    t = torch.arange(0, duration, 1.0 / sample_rate)
    sweep = torch.sin(
        (2.0 * math.pi * f0 * duration / R) * (torch.exp(t * R / duration) - 1)
    )
    if inverse:
        k = torch.exp(t * R / duration)
        inverse_filter = sweep.flip(0) / k
        return amplitude * inverse_filter

    return amplitude * sweep


def generate_reference(duration: float, sample_rate: int, decibels: float = 0, f0: float = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a reference impulse response.

    Parameters
    ----------
    duration : float
        The duration of the signal in seconds.
    sample_rate : int
        The sample rate of the signal.
    decibels : float, optional
        The amplitude of the impulse in decibels. Default is 0dB fs.
    f0 : float, optional
        The start frequency of the sweep in Hz. Default is 20.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The generated sweep tone, inverse filter, and impulse response.
    """
    amplitude = 10 ** (decibels / 20)
    f1 = sample_rate / 2

    # Generate sweep tone and inverse filter
    sweep = log_sweep_tone(sample_rate, duration, amplitude, f0, f1)
    inverse_filter = log_sweep_tone(sample_rate, duration, amplitude, f0, f1, inverse=True)
    
    # Convolution
    impulse_response = torchaudio.functional.convolve(inverse_filter, sweep, mode='full')
    impulse_response /= torch.max(torch.abs(impulse_response))

    return sweep, inverse_filter, impulse_response


def save_audio(dir_path: str, file_name: str, sample_rate: int, audio: torch.Tensor):
    output_directory = Path(dir_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{file_name}.wav"
    torchaudio.save(str(output_path), audio.unsqueeze(0), sample_rate)

    print(f"Saved {output_path}")


def plot_data(x, y, subplot, title, x_label, y_label, legend=False):
    min_length = min(len(x), len(y))
    x = x[:min_length]
    y = y[:min_length]
    subplot.plot(x, y)
    subplot.set_xlim([0, x[-1]])
    subplot.set_title(title)
    subplot.set_xlabel(x_label)
    subplot.set_ylabel(y_label)
    subplot.grid(True)
    if legend:
        subplot.legend()


def main(duration: float, sample_rate: int, audiodir: str):
    """
    Generate and save reference signals for audio processing.

    Args:
        duration (float): The duration of the reference signals in seconds.
        sample_rate (int): The sample rate of the reference signals.
        audiodir (str): The directory to save the generated audio files.

    Returns:
        None
    """
    sweep, inverse_filter, reference = generate_reference(duration, sample_rate)
    single_impulse = impulse(sample_rate, duration, decibels=-18)

    save_audio(audiodir, f"sweep_{int(sample_rate/1000)}k", sample_rate, sweep)
    save_audio(
        audiodir,
        f"inverse_filter_{int(sample_rate/1000)}k",
        sample_rate,
        inverse_filter,
    )
    save_audio(
        audiodir,
        f"generator_reference_{int(sample_rate/1000)}k",
        sample_rate,
        reference,
    )
    save_audio(
        audiodir,
        f"single_impulse_{int(sample_rate/1000)}k",
        sample_rate,
        single_impulse,
    )

    fig, ax = plt.subplots(3, 1, figsize=(15, 7))

    plot_data(
        torch.arange(0, len(sweep)) / sample_rate,
        sweep.numpy(),
        ax[0],
        "Processed Sweep Tone",
        "Time [s]",
        "Amplitude",
    )
    plot_data(
        torch.arange(0, len(inverse_filter)) / sample_rate,
        inverse_filter.numpy(),
        ax[1],
        "Inverse Filter",
        "Time [s]",
        "Amplitude",
    )
    plot_data(
        torch.arange(0, len(reference)) / sample_rate,
        reference.numpy(),
        ax[2],
        "Impulse Response",
        "Time [s]",
        "Amplitude",
    )

    fig.suptitle(f"Reference Signals - Impulse Response δ(t)")

    file_path = "reference-signals.png"
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audio files for measurements."
    )
    parser.add_argument(
        "--length", type=float, default=5.0, help="Duration of the audio files."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Sample rate of the audio files."
    )
    parser.add_argument(
        "--audiodir",
        type=str,
        default="../audio",
        help="Directory to save the audio files.",
    )
    args = parser.parse_args()

    main(args.length, args.sample_rate, args.audiodir)
