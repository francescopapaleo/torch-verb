import pytest
import torch
from torchverb.signals import impulse, sine, log_sweep_tone, generate_reference


def test_impulse():
    sample_rate = 44100
    duration = 1.0
    decibels = -0.5
    impulse_duration = 0.03

    result = impulse(sample_rate, duration, decibels, impulse_duration)

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == sample_rate * duration


def test_sine():
    sample_rate = 44100
    duration = 1.0
    amplitude = 0.5
    frequency = 440.0

    result = sine(sample_rate, duration, amplitude, frequency)

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == sample_rate * duration


def test_log_sweep_tone():
    sample_rate = 44100
    duration = 1.0
    amplitude = 0.5
    f0 = 20
    f1 = 20000
    inverse = False

    result = log_sweep_tone(sample_rate, duration, amplitude, f0, f1, inverse)

    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == sample_rate * duration


def test_generate_reference():
    duration = 1.0
    sample_rate = 44100
    decibels = 0
    f0 = 20

    sweep, inverse_filter, impulse_response = generate_reference(
        duration, sample_rate, decibels, f0
        )

    assert isinstance(sweep, torch.Tensor)
    assert isinstance(inverse_filter, torch.Tensor)
    assert isinstance(impulse_response, torch.Tensor)
    assert sweep.shape[0] == sample_rate * duration
    assert inverse_filter.shape[0] == sample_rate * duration
    assert impulse_response.shape[0] == sample_rate * duration * 2 - 1
