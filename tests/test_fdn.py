import pytest
import torch
from torchverb import FDNReverb


def test_FDNReverb_init():
    delays = [0.1, 0.2, 0.3]
    sample_rate = 44100
    mix = 0.5

    reverb = FDNReverb(delays, sample_rate, mix)

    assert isinstance(reverb, FDNReverb)
    assert reverb.sample_rate == sample_rate
    assert reverb.mix == mix
    assert all(isinstance(delay, int) for delay in reverb.delays)
    assert isinstance(reverb.feedback_gain, torch.nn.Parameter)
    assert isinstance(reverb.orthogonal_matrix, torch.nn.Parameter)
    assert all(isinstance(buffer, torch.Tensor) for buffer in reverb.delay_buffers)


def test_FDNReverb_forward():
    delays = [0.1, 0.2, 0.3]
    sample_rate = 44100
    mix = 0.5
    reverb = FDNReverb(delays, sample_rate, mix)

    input_sig = torch.rand(1, sample_rate)

    output_sig = reverb.forward(input_sig)

    assert isinstance(output_sig, torch.Tensor)
    assert output_sig.shape == input_sig.shape
