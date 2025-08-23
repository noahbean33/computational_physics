"""
Wavelet, FFT, and STFT tutorial utilities.

Refactored from a notebook export into a reusable, CLI-driven script.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalSpec:
    Fs: int = 1000
    duration: float = 1.0
    noise_std: float = 0.5
    f1: float = 10.0
    f2: float = 100.0
    change_time: float = 0.5  # seconds


# ──────────────────────────────────────────────────────────────────────────────
# Core utilities
# ──────────────────────────────────────────────────────────────────────────────

def generate_signal(spec: SignalSpec) -> tuple[np.ndarray, np.ndarray]:
    """Generate a piecewise sinusoid with noise.

    Returns (t, signal)
    """
    t = np.linspace(0, spec.duration, int(spec.Fs * spec.duration), endpoint=False)
    split_idx = int(spec.change_time * spec.Fs)
    sig1 = np.sin(2 * np.pi * spec.f1 * t[:split_idx])
    sig2 = np.sin(2 * np.pi * spec.f2 * t[split_idx:])
    signal = np.concatenate((sig1, sig2))
    if spec.noise_std > 0:
        signal = signal + spec.noise_std * np.random.randn(signal.size)
    return t, signal


def wavelet_decompose(signal: np.ndarray, wavelet: str = "db1", level: int = 4):
    return pywt.wavedec(signal, wavelet, level=level)


def wavelet_reconstruct(coeffs, wavelet: str):
    return pywt.waverec(coeffs, wavelet)


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def maybe_save_or_show(path: str | None, tight: bool = True):
    if tight:
        plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_time_signal(t: np.ndarray, signal: np.ndarray, *, save_path: str | None = None):
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal)
    plt.title("Original Signal (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    maybe_save_or_show(save_path)


def plot_fft(signal: np.ndarray, Fs: int, t: np.ndarray | None = None, *, save_path: str | None = None):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / Fs)

    plt.figure(figsize=(12, 6))
    if t is not None:
        plt.subplot(2, 1, 1)
        plt.plot(t, signal)
        plt.title("Original Signal (Time Domain)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(2, 1, 2)
    
    plt.plot(xf[: N // 2], 2.0 / N * np.abs(yf[0 : N // 2]))
    plt.title("Fourier Transform (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, Fs / 2)
    plt.grid(True)
    maybe_save_or_show(save_path)


def plot_spectrogram(signal: np.ndarray, Fs: int, nperseg: int | None = None, noverlap: int | None = None, *, save_path: str | None = None):
    if nperseg is None:
        nperseg = max(16, Fs // 10)
    if noverlap is None:
        noverlap = nperseg // 2
    f_spec, t_spec, Zxx = spectrogram(signal, Fs, nperseg=nperseg, noverlap=noverlap)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Zxx + 1e-12), shading="gouraud")
    plt.colorbar(label="Intensity (dB)")
    plt.title("Spectrogram (Time-Frequency Representation)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, Fs / 2)
    maybe_save_or_show(save_path)


def plot_wavelet_coeffs(coeffs, t: np.ndarray, level: int, change_time: float | None = None, *, save_path: str | None = None):
    plt.figure(figsize=(12, 2 + 2 * (len(coeffs))))

    titles = [f"Approximation (cA{level})"] + [f"Detail (cD{i})" for i in range(level, 0, -1)]

    # Build time-like axes per coefficient length
    coeff_time_scales = []
    for arr in coeffs:
        coeff_time_scales.append(t[: len(arr)])

    plt.subplot(len(coeffs) + 1, 1, 1)
    plt.plot(t, wavelet_reconstruct(coeffs, "db1")[: len(t)])
    plt.title("Signal (for reference)")
    plt.grid(True)
    if change_time is not None:
        plt.axvline(x=change_time, color="r", linestyle="--", label="Change")
        plt.legend()

    for i, coeff in enumerate(coeffs):
        plt.subplot(len(coeffs) + 1, 1, i + 2)
        plt.plot(coeff_time_scales[i], coeff)
        plt.title(titles[i])
        plt.xlabel("Approx. Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        if change_time is not None and i > 0:
            plt.axvline(x=change_time, color="r", linestyle="--", alpha=0.7)

    maybe_save_or_show(save_path)


def plot_reconstruction(t: np.ndarray, original: np.ndarray, coeffs, wavelet: str, *, save_path: str | None = None):
    reconstructed = wavelet_reconstruct(coeffs, wavelet)
    plt.figure(figsize=(12, 4))
    plt.plot(t, original, label="Original")
    plt.plot(t, reconstructed[: len(t)], "--", label="Reconstructed")
    plt.title("Original vs. Reconstructed Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    maybe_save_or_show(save_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wavelet / FFT / STFT tutorial script")
    p.add_argument("--Fs", type=int, default=1000, help="Sampling frequency")
    p.add_argument("--duration", type=float, default=1.0, help="Signal duration (s)")
    p.add_argument("--noise-std", type=float, default=0.5, help="Noise standard deviation")
    p.add_argument("--f1", type=float, default=10.0, help="Frequency before change (Hz)")
    p.add_argument("--f2", type=float, default=100.0, help="Frequency after change (Hz)")
    p.add_argument("--change-time", type=float, default=0.5, help="Time of frequency change (s)")

    p.add_argument("--wavelet", type=str, default="db1", help="Wavelet name for DWT")
    p.add_argument("--level", type=int, default=4, help="Decomposition level")

    p.add_argument("--outdir", type=str, default="", help="Directory to save figures. If empty, figures are shown instead.")
    p.add_argument("--prefix", type=str, default="wavelet_tutorial", help="Filename prefix for saved figures")

    p.add_argument("--no-fft", action="store_true", help="Skip FFT plot")
    p.add_argument("--no-stft", action="store_true", help="Skip spectrogram plot")
    p.add_argument("--no-wavelet", action="store_true", help="Skip wavelet decomposition plots")
    p.add_argument("--no-recon", action="store_true", help="Skip reconstruction comparison plot")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    spec = SignalSpec(
        Fs=args.Fs,
        duration=args.duration,
        noise_std=args.noise_std,
        f1=args.f1,
        f2=args.f2,
        change_time=args.change_time,
    )

    t, sig = generate_signal(spec)

    save = bool(args.outdir)
    def out(name: str) -> str | None:
        return os.path.join(args.outdir, f"{args.prefix}_{name}.png") if save else None

    # Time domain and FFT
    plot_time_signal(t, sig, save_path=out("time"))
    if not args.no_fft:
        plot_fft(sig, spec.Fs, t=t, save_path=out("fft"))

    # STFT spectrogram
    if not args.no_stft:
        plot_spectrogram(sig, spec.Fs, save_path=out("spectrogram"))

    # Wavelet decomposition and reconstruction
    if not args.no_wavelet or not args.no_recon:
        coeffs = wavelet_decompose(sig, wavelet=args.wavelet, level=args.level)
        if not args.no_wavelet:
            plot_wavelet_coeffs(coeffs, t, args.level, change_time=spec.change_time, save_path=out("wavelet_coeffs"))
        if not args.no_recon:
            plot_reconstruction(t, sig, coeffs, args.wavelet, save_path=out("reconstruction"))


if __name__ == "__main__":
    main()