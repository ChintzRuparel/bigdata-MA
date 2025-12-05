import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# BUILD THE INSANE MODEL (CNN + TRANSFORMER + LSTM)
# ============================================================

def build_insane_model(input_dim):
    inp = keras.Input(shape=(input_dim,))

    # Noise
    x = layers.GaussianNoise(0.2)(inp)

    # Expand dims (batch, features) -> (batch, 1, features)
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=1))(x)

    # CNN path
    cnn = layers.Conv1D(128, 1, activation="relu")(x)
    cnn = layers.Conv1D(256, 1, activation="relu")(cnn)
    cnn = layers.GlobalMaxPooling1D()(cnn)

    # Transformer Path
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    attn = layers.LayerNormalization()(attn)
    attn = layers.GlobalAvgPool1D()(attn)

    # LSTM Path
    lstm = layers.LSTM(128, return_sequences=True)(x)
    lstm = layers.LSTM(64)(lstm)

    # Merge all 3 paths
    z = layers.Concatenate()([cnn, attn, lstm])

    # Dense insanity
    z = layers.Dense(256, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.2)(z)

    out = layers.Dense(1)(z)

    model = keras.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ============================================================
# LOAD + CLEAN DATA (FIX FOR YOUR ERROR)
# ============================================================

df = pd.read_csv("/Users/chintzruparel/Documents/GitHub/bigdata-MA/Final Dataset/Multiset-Dataset.csv")

# ---- Fix datetime ----
if "Date/Time" in df.columns:
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
    df["timestamp"] = df["Date/Time"].astype("int64") // 1e9  # convert → seconds
    df = df.drop(columns=["Date/Time"])

# ---- Drop all text/object columns ----
non_numeric_cols = df.select_dtypes(include=["object"]).columns
print("Dropping non-numeric columns:", list(non_numeric_cols))
df = df.drop(columns=non_numeric_cols)

# ---- Drop missing values ----
df = df.dropna()


# ============================================================
# SELECT TARGET + SCALE FEATURES
# ============================================================

TARGET = "Resting Heart Rate (count/min)"   # change if needed

if TARGET not in df.columns:
    raise ValueError(f"TARGET column '{TARGET}' not found!")

X = df.drop(columns=[TARGET])
y = df[TARGET]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=True
)


# ============================================================
# TRAIN MODEL
# ============================================================

model = build_insane_model(X_train.shape[1])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)


# ============================================================
# PLOT TRAINING CURVE
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training Curve (Loss)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# ============================================================
# MAKE A PREDICTION
# ============================================================

pred = model.predict(X_test[:1]).item()
actual = float(y_test.iloc[0])

print("\n🔥 Prediction:", pred)
print("❤️ Actual:", actual)
