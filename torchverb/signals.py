import torch
import torchaudio
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
    t = torch.arange(0, duration, 1.0 / sample_rate)
    sine_wave = amplitude * torch.sin(2.0 * torch.pi * frequency * t)
    return sine_wave


def sweep_tone(
    sample_rate: int,
    duration: float,
    amplitude: float,
    f0: float = 20,
    f1: float = 20000,
    inverse: bool = False,
) -> torch.Tensor:
    R = torch.log(torch.tensor(f1 / f0))
    t = torch.arange(0, duration, 1.0 / sample_rate)
    output = torch.sin(
        (2.0 * torch.pi * f0 * duration / R) * (torch.exp(t * R / duration) - 1)
    )
    if inverse:
        k = torch.exp(t * R / duration)
        output = output.flip(0) / k
    sweep_tone = amplitude * output
    return sweep_tone


def generate_reference(
    duration: float, sample_rate: int, decibels: float = -18, f0: float = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    amplitude = 10 ** (decibels / 20)
    f1 = sample_rate / 2

    sweep = sweep_tone(sample_rate, duration, amplitude, f0=f0, f1=f1)
    inverse_filter = sweep_tone(
        sample_rate, duration, amplitude, f0=f0, f1=f1, inverse=True
    )

    reference = torchaudio.functional.convolve(inverse_filter, sweep)

    return sweep, inverse_filter, reference


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
