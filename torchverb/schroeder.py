import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt


def allpass(input_signal, delay, gain):
    output_signal = F.allpass_biquad(input_signal, delay, gain)
    return output_signal


def comb(input_signal, delay, gain):
    output_signal = F.comb_biquad(input_signal, delay, delay, gain)
    return output_signal


def comb_with_lp(input_signal, delay, g, g1):
    g2 = g * (1 - g1)
    output_signal = F.comb_biquad(input_signal, delay, delay, g1) - g2 * F.comb_biquad(input_signal, delay, delay, 1)
    return output_signal


def delay(input_signal, delay, gain=1):
    output_signal = F.delay(input_signal, delay) * gain
    return output_signal


def main():
    # Load audio using torchaudio
    sample_in = './audio/raw/plk-fm-base.wav'
    sample, sample_rate = torchaudio.load(sample_in)

    # Initialization of algorithm's variables
    stereospread = 23
    delays_r = [2205, 2469, 2690, 2998, 3175, 3439]
    delays_l = [d + stereospread for d in delays_r]
    delays_early = [877, 1561, 1715, 1825, 3082, 3510]
    gains_early = [1.02, 0.818, 0.635, 0.719, 0.267, 0.242]
    g1_list = [0.41, 0.43, 0.45, 0.47, 0.48, 0.50]
    g = 0.9
    rev_to_er_delay = 1800
    allpass_delay = 286
    allpass_g = 0.7

    output_gain = 0.075
    dry = 1
    wet = 1
    width = 1
    wet1 = wet * (width / 2 + 0.5)
    wet2 = wet * ((1 - width) / 2)

    # Convert to stereo
    sample = sample.unsqueeze(0).expand(2, -1).contiguous()

    early_reflections_r = torch.zeros_like(sample[0])
    early_reflections_l = torch.zeros_like(sample[1])
    combs_out_r = torch.zeros_like(sample[0])
    combs_out_l = torch.zeros_like(sample[1])

    # Algorithm's main part
    for i in range(6):
        early_reflections_r = early_reflections_r + delay(sample[0], delays_early[i], gains_early[i])[:sample.size(1)]
        early_reflections_l = early_reflections_l + delay(sample[1], delays_early[i], gains_early[i])[:sample.size(1)]

    for i in range(6):
        combs_out_r = combs_out_r + comb_with_lp(sample[0], delays_r[i], g, g1_list[i])
        combs_out_l = combs_out_l + comb_with_lp(sample[1], delays_l[i], g, g1_list[i])

    reverb_r = allpass(combs_out_r, allpass_delay, allpass_g)
    reverb_l = allpass(combs_out_l, allpass_delay, allpass_g)

    early_reflections_r = torch.cat((early_reflections_r, torch.zeros(rev_to_er_delay)), dim=-1)
    early_reflections_l = torch.cat((early_reflections_l, torch.zeros(rev_to_er_delay)), dim=-1)

    reverb_r = delay(reverb_r, rev_to_er_delay)
    reverb_l = delay(reverb_l, rev_to_er_delay)

    reverb_out_r = early_reflections_r + reverb_r
    reverb_out_l = early_reflections_l + reverb_l

    reverb_out_r = output_gain * ((reverb_out_r * wet1 + reverb_out_l * wet2) + torch.cat((sample[0], torch.zeros(rev_to_er_delay)), dim=-1) * dry)
    reverb_out_l = output_gain * ((reverb_out_l * wet1 + reverb_out_r * wet2) + torch.cat((sample[1], torch.zeros(rev_to_er_delay)), dim=-1) * dry)

    # Writing to file
    torchaudio.save('filename_out.wav', torch.stack((reverb_out_r, reverb_out_l)), sample_rate)

    # Plotting the results
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(sample[0].numpy())
    plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.plot(sample[1].numpy())
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(reverb_out_r.numpy())
    plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.plot(reverb_out_l.numpy())
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
