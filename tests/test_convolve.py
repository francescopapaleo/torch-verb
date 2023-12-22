import unittest
import torch
import torchaudio
from typing import Union
from torchreverb.convolve import ConvolutionReverb

class TestConvolutionReverb(unittest.TestCase):
    def setUp(self):
        self.ir_file = 'path_to_your_ir_file.wav'  # replace with a real file path
        self.conv_method = 'fft'
        self.mix = 0.5
        self.conv_reverb = ConvolutionReverb(self.ir_file, self.conv_method, self.mix)

    def test_initialization(self):
        self.assertEqual(self.conv_reverb.ir_file, self.ir_file)
        self.assertEqual(self.conv_reverb.conv_method, self.conv_method)
        self.assertEqual(self.conv_reverb.mix, self.mix / 100)

    def test_load_ir(self):
        ir_sig, ir_sr = torchaudio.load(self.ir_file)
        self.conv_reverb.load_ir()
        self.assertTrue(torch.equal(self.conv_reverb.ir_sig, ir_sig))
        self.assertEqual(self.conv_reverb.ir_sr, ir_sr)

    def test_convolve(self):
        input_sig = torch.randn(1, 44100)  # replace with a real input signal
        output_sig = self.conv_reverb.convolve(input_sig)
        self.assertEqual(output_sig.shape, input_sig.shape)

    def test_invalid_conv_method(self):
        self.conv_reverb.conv_method = 'invalid_method'
        input_sig = torch.randn(1, 44100)  # replace with a real input signal
        with self.assertRaises(ValueError):
            self.conv_reverb.convolve(input_sig)

if __name__ == '__main__':
    unittest.main()