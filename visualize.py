import numpy as np
import matplotlib.pyplot as plt
import os
from inference import run_inference

def plot_results(target, model_path, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    basename = target.replace(".fits", "").replace(" ", "_").replace("/", "_")
    
    print(f"Processing target: {target}...")
    res = run_inference(model_path, target)
    
    time, flux = res['raw_data']
    glob_view, loc_view, fft_view = res['views']
    params = res['parameters']
    
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Original Lightcurve
    ax1 = fig.add_subplot(411)
    ax1.plot(time, flux, 'k.', markersize=2, alpha=0.5)
    ax1.set_title(f"Original Light Curve - {target} (Pred: {res['predicted_class']} | Conf: {res['confidence']:.2f})")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Flux")
    
    # 2. Global View (Phase -0.5 to 0.5)
    ax2 = fig.add_subplot(412)
    ax2.plot(np.linspace(-0.5, 0.5, len(glob_view)), glob_view, 'b-')
    ax2.set_title(f"Global View (Phase Folded) | Period: {params['period']:.4f} d | SNR: {params['snr']:.2f}")
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Normalized Flux")
    
    # 3. Local View (Zoomed in on transit)
    ax3 = fig.add_subplot(413)
    ax3.plot(np.linspace(-0.1, 0.1, len(loc_view)), loc_view, 'r-')
    ax3.set_title(f"Local View (Zoomed Transit) | Depth: {params['depth']:.5f} | Dur: {params['duration']:.4f} d")
    ax3.set_xlabel("Phase")
    ax3.set_ylabel("Normalized Flux")

    # 4. Noise Fingerprint (FFT)
    ax4 = fig.add_subplot(414)
    ax4.plot(fft_view, 'g-')
    ax4.set_title("Noise Fingerprint (PSD of out-of-transit data)")
    ax4.set_xlabel("Frequency Bins")
    ax4.set_ylabel("Standardized Log Power")
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{basename}_plot.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")
    
    return res
