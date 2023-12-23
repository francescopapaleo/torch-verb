import pytest
import torch
from torchverb.echo import Echo


@pytest.fixture
def echo():
    sr = 48000
    delay = 0.5
    mix = 0.5
    return Echo(sr, delay, mix)


def test_initialization(echo):
    sr = 48000
    delay = 0.5
    mix = 0.5
    assert echo.delay_samples == round(delay * sr)
    assert echo.mix == mix


def test_delay(echo):
    sr = 48000
    input_sig = torch.randn(1, sr)  # replace with a real input signal
    output_sig = echo(input_sig)
    expected_output_shape = torch.Size([1, sr + echo.delay_samples])
    assert output_sig.shape == expected_output_shape


def test_output_signal(echo):
    sr = 48000
    input_sig = torch.randn(1, sr)  # replace with a real input signal
    output_sig = echo(input_sig)
    assert torch.all(output_sig <= 1)
    assert torch.all(output_sig >= -1)
