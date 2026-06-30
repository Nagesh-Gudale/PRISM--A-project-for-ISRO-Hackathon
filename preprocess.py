import numpy as np
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
from scipy import signal

GLOBAL_BINS = 2001
LOCAL_BINS = 201
FFT_BINS = 256

def normalize_lightcurve(flux):
    median_flux = np.nanmedian(flux)
    if median_flux == 0 or np.isnan(median_flux):
        return flux
    return (flux / median_flux) - 1.0

def remove_outliers(flux, sigma=3.0):
    median = np.nanmedian(flux)
    std = np.nanstd(flux)
    mask = np.abs(flux - median) < (sigma * std)
    return np.ma.masked_array(flux, mask=~mask)

def phase_fold(time, period, t0):
    """
    Folds time on period and centers the transit at phase 0.
    Returns phase values between -0.5 and 0.5.
    """
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    return phase

def bin_and_interpolate(phase, flux, num_bins, phase_min, phase_max):
    """
    Sorts, filters to phase window, and bins data using interpolation.
    """
    # Filter to window
    mask = (phase >= phase_min) & (phase <= phase_max)
    if not np.any(mask):
        return np.zeros(num_bins)
        
    p = phase[mask]
    f = flux[mask]
    
    # Sort by phase
    sort_idx = np.argsort(p)
    p = p[sort_idx]
    f = f[sort_idx]
    
    # Remove duplicates for interpolation
    p, unique_idx = np.unique(p, return_index=True)
    f = f[unique_idx]
    
    if len(p) < 2:
        return np.zeros(num_bins)
        
    # Interpolate onto uniform grid
    grid = np.linspace(phase_min, phase_max, num_bins)
    interpolator = interp1d(p, f, kind='linear', bounds_error=False, fill_value=0.0)
    return interpolator(grid)

def generate_global_view(phase, flux):
    """Generates the full phase curve (2001 bins)"""
    return bin_and_interpolate(phase, flux, GLOBAL_BINS, -0.5, 0.5)

def generate_local_view(phase, flux, zoom_window=0.1):
    """Generates a zoomed-in view of the transit (201 bins)"""
    return bin_and_interpolate(phase, flux, LOCAL_BINS, -zoom_window, zoom_window)

def generate_noise_fingerprint(time, flux, period, duration=None):
    """
    Generates a frequency spectrum (PSD) of the out-of-transit data.
    """
    # Simple masking of transit if duration is provided
    if duration is not None and period is not None:
        phase = ((time + 0.5 * period) % period) / period - 0.5
        phase_duration = duration / period
        oot_mask = np.abs(phase) > (phase_duration * 1.5)
        t_oot = time[oot_mask]
        f_oot = flux[oot_mask]
    else:
        t_oot, f_oot = time, flux
        
    if len(t_oot) < 100:
        return np.zeros(FFT_BINS)
        
    # Calculate Power Spectral Density (PSD) using Welch's method
    # Since sampling might be slightly irregular, we just use the raw array, 
    # assuming roughly uniform 2-min cadence for the PSD profile shape.
    frequencies, psd = signal.welch(f_oot, nperseg=min(len(f_oot), 1024))
    
    # Interpolate PSD to fixed size
    if len(frequencies) < 2:
        return np.zeros(FFT_BINS)
        
    grid = np.linspace(frequencies[0], frequencies[-1], FFT_BINS)
    interpolator = interp1d(frequencies, psd, kind='linear', bounds_error=False, fill_value=0.0)
    fingerprint = interpolator(grid)
    
    # Log scale it to handle extreme peaks and normalize
    fingerprint = np.log10(fingerprint + 1e-10)
    fingerprint = (fingerprint - np.mean(fingerprint)) / (np.std(fingerprint) + 1e-10)
    
    return fingerprint

def preprocess_for_astronet(time, flux, period, t0, duration=None):
    """
    Returns the Global View, Local View, and FFT Fingerprint.
    """
    flux_norm = normalize_lightcurve(flux)
    
    # Remove massive outliers (keeping mild red noise)
    clipped = remove_outliers(flux_norm)
    mask = ~clipped.mask
    time_c = time[mask]
    flux_c = clipped.data[mask]
    
    # Phase fold
    phase = phase_fold(time_c, period, t0)
    
    # Generate the 3 branches
    global_view = generate_global_view(phase, flux_c)
    local_view = generate_local_view(phase, flux_c)
    noise_fft = generate_noise_fingerprint(time_c, flux_c, period, duration)
    
    return global_view, local_view, noise_fft
