import pytest
import torch
from torchverb import SchroederReverb


def test_SchroederReverb_init():
    n_allpass = 4
    sample_rate = 44100
    mix = 0.5

    reverb = SchroederReverb(n_allpass, sample_rate, mix)

    assert isinstance(reverb, SchroederReverb)
    assert reverb.n_allpass == n_allpass
    assert reverb.sample_rate == sample_rate
    assert reverb.mix == mix
    assert all(isinstance(param, tuple) and len(param) == 2 for param in reverb.allpass_params)


def test_SchroederReverb_forward():
    n_allpass = 4
    sample_rate = 44100
    mix = 0.5
    reverb = SchroederReverb(n_allpass, sample_rate, mix)

    input_sig = torch.rand(1, sample_rate)

    output_sig = reverb.forward(input_sig)

    assert isinstance(output_sig, torch.Tensor)
    assert output_sig.shape == input_sig.shape
