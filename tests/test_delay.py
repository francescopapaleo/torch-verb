import pytest
import torch
from torchverb import DelayLine


@pytest.fixture
def delay():
    sr = 48000
    delays = [0.5]
    mix = 0.5
    return DelayLine(sr, delays, mix)


def test_initialization():
    sr = 48000
    delay_time = 0.5
    mix = 0.5
    delay = DelayLine(sr, [delay_time], mix)
    assert delay.delay_samples() == round(delay_time * sr)


def test_delay():
    sr = 48000
    delay_time = 0.5
    mix = 0.5
    delay = DelayLine(sr, [delay_time], mix)
    input_sig = torch.randn(1, sr)  # Replace with a real input signal
    output_sig = delay(input_sig)
    expected_output_shape = torch.Size([1, sr])
    assert output_sig.shape == expected_output_shape


def test_output_signal(delay):
    sr = 48000
    input_sig = torch.randn(1, sr)  # replace with a real input signal
    output_sig = delay(input_sig)
    assert torch.all(output_sig <= 1)
    assert torch.all(output_sig >= -1)
