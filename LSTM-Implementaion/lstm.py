import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("/Users/chintzruparel/Documents/GitHub/bigdata-MA/Final Dataset/Multiset-Dataset.csv")

TARGET = "Heart Rate [Avg] (count/min)"

df = df.select_dtypes(include=[np.number]).dropna()
if TARGET not in df:
    raise ValueError(f"{TARGET} not found in numeric columns")

# ==========================================
# 2. FAST FEATURE ENGINEERING
# ==========================================
cols = df.columns.tolist()

df_sq     = df[cols].pow(2).add_prefix("sq_")
df_cube   = df[cols].pow(3).add_prefix("cube_")
df_noise  = df[cols] * (1 + 0.01*np.random.randn(len(df), len(cols)))
df_noise  = df_noise.add_prefix("noise_")

df["roll_mean_10"] = df[TARGET].rolling(10, min_periods=1).mean()
df["roll_std_10"]  = df[TARGET].rolling(10, min_periods=1).std().fillna(0)

df = pd.concat([df, df_sq, df_cube, df_noise], axis=1)

# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================
X = df.drop(columns=[TARGET]).astype(np.float32)
y = df[TARGET].astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# ==========================================
# 4. BUILD CHAOSNET MODEL
# ==========================================
inputs = layers.Input(shape=(X_train.shape[1],))

x = layers.GaussianNoise(0.02)(inputs)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Residual block
res = layers.Dense(512, activation="relu")(x)
res = layers.Dropout(0.2)(res)
x = layers.Add()([x, res])

# Deeper layers
x = layers.Dense(512, activation="swish")(x)
x = layers.Dropout(0.4)(x)

# Fake LSTM dimension
x = layers.Reshape((1, 512))(x)
x = layers.LSTM(128)(x)

# Output
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1)(x)

model = models.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

model.summary()

# ==========================================
# 5. FIXED CYCLIC LR CALLBACK
# ==========================================
class CrazyLR(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        new_lr = 0.0001 + (np.sin(epoch/2) + 1) * 0.0005
        print(f"\n🔥 Setting new LR: {new_lr:.6f}")

        try:
            self.model.optimizer.learning_rate.assign(new_lr)
        except Exception:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)

# ==========================================
# 6. TRAIN MODEL
# ==========================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    callbacks=[
        CrazyLR(),
        callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    ]
)

# ==========================================
# 7. EVALUATE
# ==========================================
loss, mae = model.evaluate(X_test, y_test)
print("\n🔥 FINAL MAE:", mae)

# ==========================================
# 8. SAMPLE PREDICTION
# ==========================================
pred = model.predict(X_test[:1])
pred_value = pred.item()   # extracts scalar safely

print("\n🤖 Prediction:", pred_value)
print("Actual:", float(y_test.iloc[0]))

