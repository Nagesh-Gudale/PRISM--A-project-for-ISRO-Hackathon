import os
import pandas as pd
import numpy as np
import lightkurve as lk
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings('ignore')

def fetch_toi_list():
    """
    Downloads the current TESS Objects of Interest (TOI) list from ExoFOP.
    """
    toi_url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    print("Downloading the official TOI list from ExoFOP...")
    df = pd.read_csv(toi_url)
    return df

def select_balanced_subset(df, total_samples=10000):
    """
    Filters for known objects (Known/Confirmed planets) and False Positives,
    then samples them to create a balanced dataset.
    """
    half = total_samples // 2
    
    # KP = Known Planet, CP = Confirmed Planet
    known_planets = df[df['TFOPWG Disposition'].isin(['KP', 'CP'])]
    
    # FP = False Positive
    false_positives = df[df['TFOPWG Disposition'] == 'FP']
    
    # Drop rows without a valid TIC ID
    known_planets = known_planets.dropna(subset=['TIC ID'])
    false_positives = false_positives.dropna(subset=['TIC ID'])
    
    # Randomly sample
    kp_sample = known_planets.sample(n=min(half, len(known_planets)), random_state=42)
    fp_sample = false_positives.sample(n=min(half, len(false_positives)), random_state=42)
    
    kp_sample['Label'] = 'Transit'
    fp_sample['Label'] = 'False Positive'
    
    balanced_df = pd.concat([kp_sample, fp_sample])
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

def main():
    dataset_dir = "dataset"
    lc_dir = os.path.join(dataset_dir, "lightcurves")
    os.makedirs(lc_dir, exist_ok=True)
    
    # 1. Download and filter TOI list
    full_toi_df = fetch_toi_list()
    target_df = select_balanced_subset(full_toi_df, total_samples=10000)
    
    print(f"Selected {len(target_df)} objects for downloading.")
    
    downloaded_count = 0
    successful_rows = []
    
    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        tic_id = int(row['TIC ID'])
        npz_path = os.path.join(lc_dir, f"TIC_{tic_id}.npz")
        
        if os.path.exists(npz_path):
            downloaded_count += 1
            successful_rows.append(row)
            continue
            
        try:
            # Search for SPOC 2-minute cadence data
            search_result = lk.search_lightcurve(f"TIC {tic_id}", author="SPOC", exptime=120)
            
            if len(search_result) == 0:
                search_result = lk.search_lightcurve(f"TIC {tic_id}")
                
            if len(search_result) > 0:
                # Download the first available lightcurve
                lc = search_result[0].download()
                
                if lc is not None:
                    # Remove NaNs
                    lc = lc.remove_nans()
                    
                    time_arr = lc.time.value
                    flux_arr = lc.flux.value
                    
                    if hasattr(lc, 'flux_err') and lc.flux_err is not None:
                        flux_err_arr = lc.flux_err.value
                    else:
                        flux_err_arr = np.zeros_like(flux_arr)
                    
                    # Save locally
                    np.savez_compressed(npz_path, time=time_arr, flux=flux_arr, flux_err=flux_err_arr)
                    
                    downloaded_count += 1
                    successful_rows.append(row)
                    
        except Exception as e:
            # Silently pass errors (e.g. connection issues or bad data)
            pass
            
        time.sleep(0.5)

    # Save the labeled metadata for training
    success_df = pd.DataFrame(successful_rows)
    metadata_path = os.path.join(dataset_dir, "labeled_toi_dataset.csv")
    success_df.to_csv(metadata_path, index=False)
    
    print(f"\nSuccessfully downloaded {downloaded_count} lightcurves.")
    print(f"Labeled metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
