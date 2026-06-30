import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess_for_astronet, GLOBAL_BINS, LOCAL_BINS, FFT_BINS
from model import build_model
import tensorflow as tf

def load_dataset(csv_path, lc_dir):
    df = pd.read_csv(csv_path)
    X_global, X_local, X_fft, y = [], [], [], []
    
    label_map = {'Transit': 0, 'False Positive': 1}
    
    print(f"Loading and generating tri-views from {csv_path}...")
    for idx, row in df.iterrows():
        try:
            tic_id = int(row['TIC ID']) if 'TIC ID' in row else int(row['TIC_ID'])
            # Extract parameters for folding
            period = float(row['Period (days)']) if pd.notnull(row.get('Period (days)')) else None
            t0 = float(row['Epoch (BJD)']) if pd.notnull(row.get('Epoch (BJD)')) else None
            duration = float(row['Duration (hours)']) / 24.0 if pd.notnull(row.get('Duration (hours)')) else None
            
        except Exception:
            continue
            
        if period is None or t0 is None:
            # Skip if we don't have canonical parameters to fold on
            continue
            
        npz_path = os.path.join(lc_dir, f"TIC_{tic_id}.npz")
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                time = data['time']
                flux = data['flux']
                
                # Generate the 3 views
                glob_view, loc_view, fft_view = preprocess_for_astronet(time, flux, period, t0, duration)
                
                X_global.append(glob_view)
                X_local.append(loc_view)
                X_fft.append(fft_view)
                
                label_str = row['Label']
                y.append(label_map.get(label_str, 1))
            except Exception as e:
                pass
                
    X_g = np.array(X_global).reshape(-1, GLOBAL_BINS, 1)
    X_l = np.array(X_local).reshape(-1, LOCAL_BINS, 1)
    X_f = np.array(X_fft).reshape(-1, FFT_BINS, 1)
    y = np.array(y)
    
    return [X_g, X_l, X_f], y

def main():
    dataset_dir = "dataset"
    csv_path = os.path.join(dataset_dir, "labeled_toi_dataset.csv")
    lc_dir = os.path.join(dataset_dir, "lightcurves")
    
    if not os.path.exists(csv_path):
        print("Dataset not found. Run data_collector.py first.")
        return
        
    X_list, y = load_dataset(csv_path, lc_dir)
    print(f"Dataset loaded: {len(y)} samples.")
    
    if len(y) < 10:
        print("Not enough data to train. Need at least 10 valid samples with Period/Epoch.")
        return
        
    # Split the indices to ensure we split all 3 inputs identically
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = [x[train_idx] for x in X_list]
    X_val = [x[val_idx] for x in X_list]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    model = build_model(num_classes=2)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')
    
    print("Training Tri-Input model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Training complete. Best model saved to 'best_model.keras'")
    
if __name__ == "__main__":
    main()
