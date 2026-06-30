import numpy as np
import tensorflow as tf
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from preprocess import preprocess_for_astronet, GLOBAL_BINS, LOCAL_BINS, FFT_BINS, remove_outliers, normalize_lightcurve
import os

def load_lightcurve(target):
    if os.path.exists(target) and target.endswith('.fits'):
        lc = lk.read(target)
    else:
        search_result = lk.search_lightcurve(target, author="SPOC", exptime=120)
        if len(search_result) == 0:
            search_result = lk.search_lightcurve(target)
            
        if len(search_result) == 0:
            raise ValueError(f"Could not find light curve for target: {target}")
        lc = search_result[0].download()
        
    if lc is None:
        raise ValueError(f"Failed to download light curve for {target}")
        
    lc = lc.remove_nans()
    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value if hasattr(lc, 'flux_err') and lc.flux_err is not None else np.zeros_like(flux)
    
    return time, flux, flux_err

def estimate_parameters(time, flux, flux_err):
    model = BoxLeastSquares(time, flux, dy=flux_err)
    period_grid = np.linspace(0.5, 20.0, 1000)
    duration_grid = np.linspace(0.01, 0.2, 50) 
    
    results = model.power(period_grid, duration_grid)
    
    best_idx = np.argmax(results.power)
    best_period = results.period[best_idx]
    best_duration = results.duration[best_idx]
    best_depth = results.depth[best_idx]
    t0 = results.transit_time[best_idx]
    
    transit_mask = model.transit_mask(time, best_period, best_duration, t0)
    oot_flux = flux[~transit_mask]
    noise_level = np.nanstd(oot_flux) if len(oot_flux) > 0 else 1e-5
    snr = best_depth / noise_level if noise_level > 0 else 0
    
    return best_period, t0, best_duration, best_depth, snr

def run_inference(model_path, target):
    # 1. Load Raw Data
    time, flux, flux_err = load_lightcurve(target)
    
    # 2. Run BLS FIRST (Astronet requirement)
    # We use a cleaned version for BLS
    norm_flux = normalize_lightcurve(flux)
    cleaned = remove_outliers(norm_flux)
    valid = ~cleaned.mask
    time_c = time[valid]
    flux_c = cleaned.data[valid]
    flux_err_c = flux_err[valid]
    
    period, t0, duration, depth, snr = estimate_parameters(time_c, flux_c, flux_err_c)
    
    # 3. Generate Tri-Views
    glob_view, loc_view, fft_view = preprocess_for_astronet(time, flux, period, t0, duration)
    
    # 4. CNN Classification
    model = tf.keras.models.load_model(model_path)
    X_g = glob_view.reshape(1, GLOBAL_BINS, 1)
    X_l = loc_view.reshape(1, LOCAL_BINS, 1)
    X_f = fft_view.reshape(1, FFT_BINS, 1)
    
    preds = model.predict([X_g, X_l, X_f], verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    
    classes = {0: "Planet Candidate (Transit)", 1: "False Positive/Noise"}
    predicted_class = classes.get(class_idx, "Unknown")
    
    results = {
        'target': target,
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'parameters': {
            'period': period,
            't0': t0,
            'duration': duration,
            'depth': depth,
            'snr': snr
        },
        'raw_data': (time, flux),
        'views': (glob_view, loc_view, fft_view)
    }
    
    return results
