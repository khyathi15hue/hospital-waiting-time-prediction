"""
Trains the hybrid LSTM model (queueing-theory features + recent wait-time
sequence) and exports everything the backend needs:
  - lstm_hybrid_model.keras   (trained model)
  - encoders.json             (category -> integer mappings)
  - static_scaler.json        (mean/scale for static features, for live scaling)
  - seq_scaler.json           (mean/scale for the wait-time sequence)
  - department_baselines.json (consult-duration averages, used for the M/M/c estimate)
  - department_wait_seed.json (last N actual wait times per department, to seed
                                the live rolling history instead of starting at zero)

Usage:
    python export_artifacts.py path/to/Hospital_Wait__TIme_Data.csv
"""
import sys, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "Hospital_Wait__TIme_Data.csv"
TARGET = "TriageToProviderStartTime"
WINDOW = 5

CAT_FEATURES = ["AgeGroup", "Department", "AppointmentType", "InsuranceType", "ArrivalMethod",
    "TriageCategory", "ProviderID", "DayOfWeek", "BookingType"]
NUM_FEATURES = ["FacilityOccupancyRate", "ProvidersOnShift", "NursesOnShift", "StaffToPatientRatio",
    "IsOnlineBooking", "ArrivalHour", "ArrivalDayOfWeek", "IsWeekend", "QueueLengthAhead", "MMc_estimate"]
STATIC_COLS = CAT_FEATURES + NUM_FEATURES

print("Loading data...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TARGET]).copy()
df["CheckInTime_dt"] = pd.to_datetime(df["CheckInTime"], format="%d-%m-%Y %H:%M")
df["ProviderStartTime_dt"] = pd.to_datetime(df["ProviderStartTime"], format="%d-%m-%Y %H:%M")
df = df.sort_values("CheckInTime_dt").reset_index(drop=True)

print("Reconstructing queue lengths (queueing-theory features)...")
queue_lengths = []
for dept, group in df.groupby("Department"):
    group = group.sort_values("CheckInTime_dt")
    checkins = group["CheckInTime_dt"].values
    starts = group["ProviderStartTime_dt"].values
    idx = group.index.values
    for i in range(len(group)):
        ahead = np.sum((checkins < checkins[i]) & (starts > checkins[i]))
        queue_lengths.append((idx[i], ahead))
queue_df = pd.DataFrame(queue_lengths, columns=["orig_idx", "QueueLengthAhead"]).set_index("orig_idx")
df = df.join(queue_df)
dept_avg_service_series = df.groupby("Department")["ConsultationDurationTime"].mean()
df["MMc_estimate"] = df["QueueLengthAhead"] * (
    df["Department"].map(dept_avg_service_series) / df["ProvidersOnShift"]
)

df_model = df.dropna(subset=STATIC_COLS + [TARGET]).copy().reset_index(drop=True)
print(f"Training on {len(df_model)} rows")

print("Building wait-time sequences...")
sequences = np.zeros((len(df_model), WINDOW, 1), dtype=float)
dept_wait_seed = {}
for dept, group in df_model.groupby("Department"):
    group = group.sort_values("CheckInTime_dt")
    wait_times = group[TARGET].values
    idxs = group.index.values
    for pos, row_idx in enumerate(idxs):
        history = wait_times[max(0, pos - WINDOW):pos]
        if len(history) > 0:
            sequences[row_idx, WINDOW - len(history):, 0] = history
    last_n = wait_times[-WINDOW:].tolist()
    dept_wait_seed[dept] = [0.0] * (WINDOW - len(last_n)) + last_n

encoders = {}
for col in CAT_FEATURES:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = list(le.classes_)

X_static = df_model[STATIC_COLS].values.astype(float)
y = df_model[TARGET].values.astype(float)

static_scaler = StandardScaler()
X_static_scaled = static_scaler.fit_transform(X_static)
seq_scaler = StandardScaler()
seq_scaled = seq_scaler.fit_transform(sequences.reshape(-1, 1)).reshape(sequences.shape)

indices = np.arange(len(df_model))
idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

seq_input = keras.Input(shape=(WINDOW, 1), name="wait_history")
x1 = layers.LSTM(32)(seq_input)
static_input = keras.Input(shape=(X_static.shape[1],), name="static_features")
x2 = layers.Dense(32, activation="relu")(static_input)
merged = layers.Concatenate()([x1, x2])
merged = layers.Dense(32, activation="relu")(merged)
merged = layers.Dropout(0.2)(merged)
output = layers.Dense(1)(merged)
model = keras.Model(inputs=[seq_input, static_input], outputs=output)
model.compile(optimizer="adam", loss="mae")

early_stop = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
model.fit(
    [seq_scaled[idx_train], X_static_scaled[idx_train]], y[idx_train],
    validation_split=0.15, epochs=100, batch_size=32,
    callbacks=[early_stop], verbose=0,
)

preds = model.predict([seq_scaled[idx_test], X_static_scaled[idx_test]], verbose=0).flatten()
mae = mean_absolute_error(y[idx_test], preds)
print(f"\nLSTM hybrid test MAE: {mae:.2f} minutes")

model.save("lstm_hybrid_model.keras")
with open("encoders.json", "w") as f:
    json.dump(encoders, f, indent=2)
with open("static_scaler.json", "w") as f:
    json.dump({"mean": static_scaler.mean_.tolist(), "scale": static_scaler.scale_.tolist(),
               "columns": STATIC_COLS}, f, indent=2)
with open("seq_scaler.json", "w") as f:
    json.dump({"mean": seq_scaler.mean_.tolist(), "scale": seq_scaler.scale_.tolist()}, f, indent=2)

dept_baselines = {}
for dept in df["Department"].unique():
    rows = df[df["Department"] == dept].sort_values("CheckInTime_dt")
    span_hours = (rows["CheckInTime_dt"].max() - rows["CheckInTime_dt"].min()).total_seconds() / 3600
    avg_arrival_rate_per_hour = (len(rows) - 1) / span_hours if span_hours > 0 else 1.0
    dept_baselines[dept] = {
        "avg_consult_minutes": float(rows["ConsultationDurationTime"].mean()),
        "typical_providers_on_shift": int(round(rows["ProvidersOnShift"].mean())),
        "avg_arrival_rate_per_hour": round(float(avg_arrival_rate_per_hour), 3),
        "avg_staff_to_patient_ratio": float(rows["StaffToPatientRatio"].mean()),
    }
with open("department_baselines.json", "w") as f:
    json.dump(dept_baselines, f, indent=2)
with open("department_wait_seed.json", "w") as f:
    json.dump(dept_wait_seed, f, indent=2)

print("\nExported: lstm_hybrid_model.keras, encoders.json, static_scaler.json,")
print("          seq_scaler.json, department_baselines.json, department_wait_seed.json")
