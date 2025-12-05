import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


# ===============================================================
# LOAD + CLEAN DATA
# ===============================================================

df = pd.read_csv("/Users/chintzruparel/Documents/GitHub/bigdata-MA/Final Dataset/Multiset-Dataset.csv").dropna()

# Ensure Date/Time is parsed
df["Date/Time"] = pd.to_datetime(df["Date/Time"])
df["timestamp"] = df["Date/Time"].astype("int64") // 1e9
df = df.drop(columns=["Date/Time"])

# Keep only numeric fields
df = df.select_dtypes(include=[np.number]).copy()

# Save a copy of timestamp for plotting, but do NOT scale it as a feature
timestamps = df["timestamp"].values
feature_df = df.drop(columns=["timestamp"])

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(feature_df).astype("float32")

input_dim = X.shape[1]
latent_dim = 3


# ===============================================================
# ENCODER
# ===============================================================

inputs = keras.Input(shape=(input_dim,))
h = layers.Dense(128, activation="relu")(inputs)
h = layers.Dense(64, activation="relu")(h)
h = layers.Dense(32, activation="relu")(h)

z_mean = layers.Dense(latent_dim, name="z_mean")(h)
z_logvar = layers.Dense(latent_dim, name="z_logvar")(h)


def sampler(args):
    m, lv = args
    eps = tf.random.normal(shape=tf.shape(m))
    return m + tf.exp(0.5 * lv) * eps


z = layers.Lambda(sampler, output_shape=(latent_dim,), name="z")([z_mean, z_logvar])

encoder = keras.Model(inputs, [z_mean, z_logvar, z], name="encoder")


# ===============================================================
# DECODER
# ===============================================================

latent_inputs = keras.Input(shape=(latent_dim,))
d = layers.Dense(64, activation="relu")(latent_inputs)
d = layers.Dense(128, activation="relu")(d)
decoded = layers.Dense(input_dim, name="decoder_output")(d)

decoder = keras.Model(latent_inputs, decoded, name="decoder")


# ===============================================================
# CUSTOM VAE MODEL (KERAS 3 + TF 2.15 COMPATIBLE)
# ===============================================================

class EXO_VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        # data is X (already float32)
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(data, training=True)
            recon = self.decoder(z, training=True)

            # Reconstruction loss (MSE)
            recon_loss = tf.reduce_mean(tf.square(data - recon))

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )

            loss = recon_loss + 0.01 * kl_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def call(self, inputs, training=False):
        z_mean, z_logvar, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)


vae = EXO_VAE(encoder, decoder)
vae.compile(optimizer="adam")


# ===============================================================
# TRAIN
# ===============================================================

print("\n🚀 Training EXO Variational Autoencoder...\n")
vae.fit(
    X,
    epochs=40,
    batch_size=32,
    verbose=1
)

# ===============================================================
# LATENT REPRESENTATION + CHAOS METRIC
# ===============================================================

z_mean_val, z_logvar_val, Z = encoder.predict(X)

# Small perturbation in input space to probe sensitivity
eps = np.random.normal(0, 0.01, size=X.shape).astype("float32")
z_mean_p, z_logvar_p, Z_perturbed = encoder.predict(X + eps)

# Lyapunov-like chaos metric: norm between latent states
EXO_chaos = np.linalg.norm(Z - Z_perturbed, axis=1)


# ===============================================================
# BASELINE VIA GAUSSIAN MIXTURE ON "GOOD" DAYS
# ===============================================================

# Use lowest resting HR as proxy for "good" physiology
rhr_col = "Resting Heart Rate (count/min)"
if rhr_col not in feature_df.columns:
    raise ValueError(f"Column '{rhr_col}' not found in health_data.csv")

# good_days = 30 calmest days
good_idx = feature_df[rhr_col].nsmallest(30).index
Z_good = Z[good_idx]

gmm = GaussianMixture(n_components=3, random_state=42).fit(Z_good)
EXO_baseline = -gmm.score_samples(Z)  # higher = more "anomalous"


# ===============================================================
# FINAL EXO INDEX (0–100)
# ===============================================================

alpha = 1.2   # chaos weight
beta = 0.8    # novelty/anomaly weight

EXO_raw = np.exp(-(alpha * EXO_chaos + beta * EXO_baseline))
EXO = 100 * (EXO_raw - EXO_raw.min()) / (EXO_raw.max() - EXO_raw.min() + 1e-8)

df["EXO_Index"] = EXO


# ===============================================================
# HELPER FOR SCALING SERIES (FOR PLOTTING ONLY)
# ===============================================================

def scale_to_0_100(series: pd.Series) -> np.ndarray:
    s = series.to_numpy(dtype=float)
    return 100 * (s - s.min()) / (s.max() - s.min() + 1e-8)


# ===============================================================
# PLOT EXO VS SCREEN TIME, RHR, FEELS LIKE TEMP
# ===============================================================

plt.figure(figsize=(15, 7))

# EXO
plt.plot(timestamps, df["EXO_Index"], label="EXO Index", color="magenta", linewidth=3)

# Screen time (screen_on column)
if "screen_on" in feature_df.columns:
    scr_scaled = scale_to_0_100(feature_df["screen_on"])
    plt.plot(timestamps, scr_scaled, label="Screen On (scaled)", alpha=0.6)

# Resting heart rate
rhr_scaled = scale_to_0_100(feature_df[rhr_col])
plt.plot(timestamps, rhr_scaled, label="Resting HR (scaled)", alpha=0.6)

# Feels like temperature
if "feels_like" in feature_df.columns:
    temp_scaled = scale_to_0_100(feature_df["feels_like"])
    plt.plot(timestamps, temp_scaled, label="Feels Like Temp (scaled)", alpha=0.6)

plt.legend()
plt.title("EXO Index vs Digital Load, Cardiovascular & Environmental Signals")
plt.xlabel("Time")
plt.ylabel("Index / Scaled (0–100)")
plt.grid(True)
plt.tight_layout()
plt.show()
