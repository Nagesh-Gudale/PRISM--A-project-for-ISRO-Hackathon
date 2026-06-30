import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from preprocess import GLOBAL_BINS, LOCAL_BINS, FFT_BINS

def build_model(num_classes=2):
    """
    Builds a Tri-Input CNN (Astronet + Noise Fingerprint) with Heavy Regularization.
    """
    # --- 1. Global Branch ---
    input_global = Input(shape=(GLOBAL_BINS, 1), name="global_input")
    x = Conv1D(16, kernel_size=5, activation='relu', padding='same')(input_global)
    x = BatchNormalization()(x)
    x = Conv1D(16, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Flatten()(x)
    global_features = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)

    # --- 2. Local Branch ---
    input_local = Input(shape=(LOCAL_BINS, 1), name="local_input")
    y = Conv1D(16, kernel_size=5, activation='relu', padding='same')(input_local)
    y = BatchNormalization()(y)
    y = Conv1D(16, kernel_size=5, activation='relu', padding='same')(y)
    y = BatchNormalization()(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Conv1D(32, kernel_size=5, activation='relu', padding='same')(y)
    y = BatchNormalization()(y)
    y = Conv1D(32, kernel_size=5, activation='relu', padding='same')(y)
    y = BatchNormalization()(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Flatten()(y)
    local_features = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(y)

    # --- 3. Noise Fingerprint Branch (FFT) ---
    input_fft = Input(shape=(FFT_BINS, 1), name="fft_input")
    z = Conv1D(16, kernel_size=5, activation='relu', padding='same')(input_fft)
    z = BatchNormalization()(z)
    z = MaxPooling1D(pool_size=2)(z)
    z = Conv1D(32, kernel_size=5, activation='relu', padding='same')(z)
    z = BatchNormalization()(z)
    z = MaxPooling1D(pool_size=2)(z)
    z = Flatten()(z)
    fft_features = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(z)

    # --- Combine Features ---
    combined = concatenate([global_features, local_features, fft_features])
    
    # --- Final Dense Layers ---
    d = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(combined)
    d = Dropout(0.5)(d)
    d = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(d)
    d = Dropout(0.5)(d)
    d = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(d)
    
    output = Dense(num_classes, activation='softmax', name="output")(d)
    
    model = Model(inputs=[input_global, input_local, input_fft], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
