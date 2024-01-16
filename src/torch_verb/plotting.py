"""
Copyright (C) 2024 Francesco Papaleo
Distributed under the GNU Affero General Public License v3.0
"""

import matplotlib.pyplot as plt


def plot_reference_ir():
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))

    # Plot the sweep tone
    axs[0].plot(sweep.numpy())  # Convert tensor to numpy array for plotting
    axs[0].set_title("Sweep Tone")
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("Amplitude")

    # Plot the inverse filter
    axs[1].plot(inverse_filter.numpy())
    axs[1].set_title("Inverse Filter")
    axs[1].set_xlabel("Sample")
    axs[1].set_ylabel("Amplitude")

    # Plot the impulse response
    axs[2].plot(impulse_response.numpy())
    axs[2].set_title("Impulse Response")
    axs[2].set_xlabel("Sample")
    axs[2].set_ylabel("Amplitude")

    # Adjust the layout
    plt.tight_layout()
    plt.show()


def plot_waterfall(waveform, title, sample_rate, args, stride=1):
    frequencies, times, Sxx = signal.spectrogram(
        waveform,
        fs=sample_rate,
        window="blackmanharris",
        nperseg=32,
        noverlap=16,
        scaling="spectrum",
        mode="magnitude",
    )
    # Convert magnitude to dB
    Sxx_dB = 20 * np.log10(Sxx)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    X, Y = np.meshgrid(frequencies, times[::stride])
    Z = Sxx_dB.T[::stride]

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="inferno",
        edgecolor="none",
        alpha=0.8,
        linewidth=0,
        antialiased=False,
    )

    # Autoscale and add colorbar
    ax.autoscale()
    cbar = fig.colorbar(surf, ax=ax, pad=0.01, aspect=35, shrink=0.5)
    cbar.set_label("Magnitude (dB)")

    # Set labels and title
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (seconds)")
    ax.set_zlabel("Magnitude (dB)")

    ax.set_xlim([frequencies[-1], frequencies[0]])
    ax.view_init(
        elev=10, azim=45, roll=None, vertical_axis="z"
    )  # Adjusts the viewing angle for better visualization
    plt.tight_layout()
    plt.savefig(title)


def plot_ir_spectrogram(signal, sample_rate, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    duration_seconds = len(signal) / sample_rate

    cax = ax.specgram(
        signal,
        NFFT=512,
        Fs=sample_rate,
        noverlap=256,
        cmap="hot",
        scale="dB",
        mode="magnitude",
        vmin=-100,
        vmax=0,
    )
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel(f"Time [sec] ({duration_seconds:.2f} s)")  # Label in seconds
    ax.grid(True)
    ax.set_title(title)

    cbar = fig.colorbar(mappable=cax[3], ax=ax, format="%+2.0f dB")
    cbar.set_label("Intensity [dB]")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved spectrogram plot to {save_path}")
