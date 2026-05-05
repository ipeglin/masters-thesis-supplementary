import matplotlib.pyplot as plt
import numpy as np

from lib.fs.get_script_name import get_name
from lib.fs.project_config import REPO_ROOT
from plotting import plot_config
from plotting.plot_config import get_figs_output_dir


def plot_morlet_wavelet():
    t = np.linspace(-4, 4, 1000)

    # Exact match with Rust complex_morlet implementation
    center_frequency = 6.0
    bandwidth = 1.0
    scale = 1.0

    norm = 1.0 / np.sqrt(np.pi * bandwidth**2 * scale**2)
    correction = np.exp(-(center_frequency**2) / (2.0 * bandwidth**2))

    gauss = np.exp(-(t**2) / (2.0 * bandwidth**2))
    exp_term = np.exp(1j * center_frequency * t)
    dc_correction = exp_term - correction if correction > 1e-10 else exp_term

    wavelet = norm * gauss * dc_correction
    real_part = np.real(wavelet)
    imag_part = np.imag(wavelet)
    envelope = norm * gauss

    # plt.figure(figsize=(10, 6))
    plt.figure()
    plt.plot(t, real_part, label="Real Part (Cosine)")
    plt.plot(t, envelope, label="Gaussian Envelope", c="k", linestyle=":")
    plt.plot(t, -envelope, linestyle=":")

    plt.title("Morlet Wavelet")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.savefig(get_figs_output_dir() / "morlet.pdf")

    # fig = plt.figure(figsize=(10, 8))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(t, real_part, imag_part, label="Complex Wavelet")
    ax.set_title("Complex Morlet Wavelet (3D)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Real Part")
    ax.set_zlabel("Imaginary Part")
    ax.legend()
    plt.savefig(get_figs_output_dir() / "complex_morlet_3d.pdf")
    # plt.show()


if __name__ == "__main__":
    plot_morlet_wavelet()
