import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt

# ==========================================================
# 1. LOAD + PREP DATA
# ==========================================================

# Expect health_data.csv in same directory
df = pd.read_csv("/Users/chintzruparel/Documents/GitHub/bigdata-MA/Final Dataset/Multiset-Dataset.csv").dropna()

# Parse time for plotting, but don't use it as a feature
df["Date/Time"] = pd.to_datetime(df["Date/Time"])
timestamps = df["Date/Time"].astype("int64") // 10**9  # seconds since epoch

# Keep only numeric columns as features (drop the datetime)
feature_df = df.select_dtypes(include=[np.number]).copy()

# Convert to numpy
X = feature_df.to_numpy().astype("float32")
num_features = X.shape[1]

print(f"Loaded data with shape: {X.shape} (samples, features)")

# ==========================================================
# 2. CUSTOM SIREN SINE ACTIVATION
# ==========================================================

@keras.utils.register_keras_serializable()
def sine_activation(x):
    return tf.sin(x)


# ==========================================================
# 3. HYPERDIMENSIONAL PROJECTION (HDP)
# ==========================================================

def HD_Projection(dim=4096):
    return keras.Sequential(
        [
            layers.Dense(dim, activation=sine_activation),
            layers.Dense(dim, activation=sine_activation),
            layers.Dense(dim, activation=None),
        ],
        name="HD_Proj",
    )

hd_proj = HD_Projection(dim=4096)


# ==========================================================
# 4. ODE BLOCK + TRANSFORMER-LIKE LATENT DYNAMICS
# ==========================================================

class ODEBlock(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(units, activation="tanh")
        self.dense2 = layers.Dense(units)

    def call(self, x):
        dx = self.dense2(self.dense1(x))
        # Small Euler integration step
        return x + 0.1 * dx


def EXO_Transformer(hidden=512, heads=8):
    inp = keras.Input((4096,), name="exo_tr_input")

    # Treat this as a 1-step "sequence" for attention
    x = layers.Reshape((1, 4096))(inp)

    # Multi-head self-attention over feature space
    x_attn = layers.MultiHeadAttention(num_heads=heads, key_dim=64)(
        x, x
    )
    x = layers.Add()([x, x_attn])
    x = layers.LayerNormalization()(x)

    # MLP block
    mlp = layers.Dense(hidden, activation="gelu")(x)
    mlp = layers.Dense(4096)(mlp)
    x = layers.Add()([x, mlp])

    # ODE dynamics in feature-latent space
    x = ODEBlock(4096, name="latent_ode")(x)

    # Flatten back to vector
    x = layers.Flatten()(x)

    return keras.Model(inp, x, name="EXO_Transformer")

exo_tr = EXO_Transformer()


# ==========================================================
# 5. CROSS-MODAL FUSION + SCALAR HEAD
# ==========================================================

def FusionBlock():
    inp = keras.Input((4096,), name="fusion_input")
    x = layers.Dense(1024, activation="gelu")(inp)
    x = layers.Dense(256, activation="gelu")(x)
    x = layers.Dense(64, activation="gelu")(x)
    return keras.Model(inp, x, name="Fusion")

fusion = FusionBlock()

psi_head = keras.Sequential(
    [
        layers.Dense(32, activation="gelu"),
        layers.Dense(1, activation="sigmoid"),  # EXO-Ψ in [0, 1]
    ],
    name="EXO_Psi_Head",
)


# ==========================================================
# 6. EXO-HYPERMIND MODEL (SUBCLASSED, SELF-SUPERVISED)
# ==========================================================

class EXO_HYPERMIND(keras.Model):
    def __init__(self, proj, trans, fuse, psi, **kwargs):
        super().__init__(**kwargs)
        self.proj = proj
        self.trans = trans
        self.fuse = fuse
        self.psi = psi

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data):
        # Keras may pass (x, y) even if we don't give y; handle robustly
        if isinstance(data, tuple) or isinstance(data, list):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            # Hyperdimensional projection
            hd = self.proj(x, training=True)

            # Transformer-like latent dynamics
            z = self.trans(hd, training=True)

            # Fusion head
            fused = self.fuse(z, training=True)

            # Scalar EXO-Ψ prediction
            psi = self.psi(fused, training=True)

            # -------- Self-supervised temporal-like losses within batch --------
            batch_size = tf.shape(x)[0]

            def compute_smooth_and_chaos():
                # Smoothness in input space (encourage continuity)
                smooth = tf.reduce_mean(tf.square(x[1:] - x[:-1]))

                # Chaos / sensitivity in latent space (encourage expressivity)
                chaos = tf.reduce_mean(tf.abs(z[1:] - z[:-1]))
                return smooth, chaos

            smooth, chaos = tf.cond(
                tf.greater(batch_size, 1),
                lambda: compute_smooth_and_chaos(),
                lambda: (tf.constant(0.0, dtype=tf.float32),
                         tf.constant(0.0, dtype=tf.float32)),
            )

            # Encourage non-trivial psi distribution
            psi_mean = tf.reduce_mean(psi)

            # Total loss: chaos (want some), smoothness (regularize), psi spread
            loss = chaos + 0.1 * smooth + 0.5 * psi_mean

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": loss,
            "smooth_loss": smooth,
            "chaos_loss": chaos,
            "psi_mean": psi_mean,
        }

    def call(self, x, training=False):
        hd = self.proj(x, training=training)
        z = self.trans(hd, training=training)
        fused = self.fuse(z, training=training)
        return self.psi(fused, training=training)


# ==========================================================
# 7. INSTANTIATE + TRAIN
# ==========================================================

model = EXO_HYPERMIND(hd_proj, exo_tr, fusion, psi_head)
model.compile(optimizer=keras.optimizers.Adam(1e-4))

print("\n🔥 TRAINING EXO-HYPERMIND...\n")
model.fit(X, epochs=25, batch_size=32, verbose=1)


# ==========================================================
# 8. COMPUTE EXO-Ψ INDEX (0–100)
# ==========================================================

exo_raw = model.predict(X).flatten()
EXO_PSI = 100.0 * (exo_raw - exo_raw.min()) / (exo_raw.max() - exo_raw.min() + 1e-8)

df["EXO_PSI"] = EXO_PSI

print("\nSample of EXO_PSI values:")
print(df["EXO_PSI"].head())


# ==========================================================
# 9. PLOT EXO-Ψ OVER TIME
# ==========================================================

plt.figure(figsize=(16, 7))
plt.plot(timestamps, EXO_PSI, label="EXO-Ψ Index", linewidth=3, color="purple")
plt.title("EXO-Ψ: Hyperdimensional Transformer Human Stability Index")
plt.xlabel("Time")
plt.ylabel("Index (0–100)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
