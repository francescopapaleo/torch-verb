import pytest
import torch
from torchverb import DelayLine


@pytest.fixture
def delay():
    sr = 48000
    delay = 0.5
    mix = 0.5
    return DelayLine(sr, delay, mix)


def test_initialization(delay):
    sr = 48000
    delay = 0.5
    mix = 0.5
    assert delay.delay_samples == round(delay * sr)
    assert delay.mix == mix


def test_delay(delay):
    sr = 48000
    input_sig = torch.randn(1, sr)  # replace with a real input signal
    output_sig = delay(input_sig)
    expected_output_shape = torch.Size([1, sr + delay.delay_samples])
    assert output_sig.shape == expected_output_shape


def test_output_signal(delay):
    sr = 48000
    input_sig = torch.randn(1, sr)  # replace with a real input signal
    output_sig = delay(input_sig)
    assert torch.all(output_sig <= 1)
    assert torch.all(output_sig >= -1)
