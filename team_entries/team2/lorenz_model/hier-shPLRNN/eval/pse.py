import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_and_smooth_power_spectrum(x, smoothing=1.0):
    """Compute and smooth the power spectrum of a time series.
    Args:
        x: Time series. Shape (S, T, dz). Spectrum taken over T.
        smoothing: gaussian filter width."""
    if x.ndim == 2:
        x = x[None]
    x_ =  (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    fft_real = np.fft.rfft(x_.cpu(), axis=1)
    ps = np.abs(fft_real)**2 * 2 / len(x_)
    if smoothing > 0:
        ps = gaussian_filter1d(ps, smoothing, axis=1)
    return ps / ps.sum(axis=1, keepdims=True)

def hellinger_distance(p, q):
    """Hellinger distance between two power spectra.
    Args:
        p, q: Power spectra. Shape (S, w, dz)."""
    # return np.sqrt(1 - np.sum(np.sqrt(p * q), axis=1))
    return np.sqrt(np.abs(1 - np.sum(np.sqrt(p * q), axis=1)))

def power_spectrum_error(p, q):
    """Power spectrum error between two power spectra.
    Simply averages the hellinger distance over the observations.
    Args:
        p, q: Power spectra. Shape (S, w, dz)."""
    return hellinger_distance(p, q).mean(axis=-1)