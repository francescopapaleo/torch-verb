import torch
import pytest
from pathlib import Path
from torchverb import ConvolutionReverb


@pytest.fixture
def IR_FILE():
    return Path("./audio/ir_analog/IR_AKG_BX25_1500ms_48kHz24b.wav")


def test_ConvolutionReverb_init(IR_FILE):
    reverb = ConvolutionReverb(IR_FILE)
    assert isinstance(
        reverb, ConvolutionReverb
    ), "Object is not an instance of ConvolutionReverb"


def test_ConvolutionReverb_process(IR_FILE):
    reverb = ConvolutionReverb(IR_FILE)
    sr = 48000
    input_sig = torch.randn(1, sr * 2)  # replace with a real input signal
    output_sig = reverb(input_sig)
    assert (
        output_sig.shape == input_sig.shape
    ), "Output signal shape does not match input signal shape"


@pytest.mark.parametrize("input_sig", [torch.randn(1, 48000), torch.randn(1, 24000)])
def test_ConvolutionReverb_process_with_different_inputs(IR_FILE, input_sig):
    reverb = ConvolutionReverb(IR_FILE)
    output_sig = reverb(input_sig)
    assert (
        output_sig.shape == input_sig.shape
    ), "Output signal shape does not match input signal shape"
