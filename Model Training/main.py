import pandas as pd
import glob
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import os

# Define the folder where all your CSVs are stored
folder_path = "dataset/"  # Adjust this to your actual folder

# Load all CSV files with stripped column names
print("Loading data from CSV files...")
all_files = glob.glob(folder_path + "*.csv")
if not all_files:
    raise FileNotFoundError(f"No CSV files found in {folder_path}")

df_list = []
for file in all_files:
    try:
        print(f"Reading file: {file}")
        df = pd.read_csv(file, skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        df_list.append(df)
        print(f"Successfully loaded {len(df)} rows from {file}")
    except Exception as e:
        print(f"Error reading {file}: {str(e)}")
        continue

if not df_list:
    raise ValueError("No data was successfully loaded from any files")

# Concatenate all dataframes
print("\nConcatenating all dataframes...")
data = pd.concat(df_list, ignore_index=True)
print(f"Total rows after concatenation: {len(data)}")

# Filter out invalid data
data = data[data["CurrentLapTime"] >= 0]  # Filter out negative CurrentLapTime
print(f"Rows after filtering invalid CurrentLapTime: {len(data)}")

# Preprocess the target columns
print("\nPreprocessing target columns...")
data["Acceleration"] = (data["Acceleration"] > 0.5).astype(int)  # Binary: 0 or 1
data["Braking"] = (data["Braking"] > 0.5).astype(int)  # Binary: 0 or 1
data["Gear"] = np.clip(
    data["Gear"].round().astype(int), 1, 6
)  # Integer between 1 and 6
data["Steering"] = np.clip(data["Steering"], -1, 1)  # Steering in range [-1, 1]
print("Preprocessing complete")

# Define selected input columns (37 features)
input_columns = [
    "Track_1",
    "Track_2",
    "Track_3",
    "Track_4",
    "Track_5",
    "Track_6",
    "Track_7",
    "Track_8",
    "Track_9",
    "Track_10",
    "Track_11",
    "Track_12",
    "Track_13",
    "Track_14",
    "Track_15",
    "Track_16",
    "Track_17",
    "Track_18",
    "Track_19",
    "SpeedX",
    "SpeedY",
    "SpeedZ",
    "Angle",
    "TrackPosition",
    "RPM",
    "WheelSpinVelocity_1",
    "WheelSpinVelocity_2",
    "WheelSpinVelocity_3",
    "WheelSpinVelocity_4",
    "DistanceCovered",
    "DistanceFromStart",
    "CurrentLapTime",
    "Damage",
    "Opponent_9",
    "Opponent_10",
    "Opponent_11",
    "Opponent_19",
]

# Define target columns
target_columns = ["Steering", "Acceleration", "Braking"]

# Extract input (X) and target (y) data
X = data[input_columns]
y = data[target_columns].copy()

# Scale only Steering target
steering_scaler = MinMaxScaler(feature_range=(-1, 1))
y[["Steering"]] = steering_scaler.fit_transform(y[["Steering"]])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the neural network
model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    solver="adam",
    max_iter=10000,
    tol=1e-5,
    early_stopping=False,
    batch_size=64,
    learning_rate="adaptive",
    learning_rate_init=0.001,
    alpha=0.0001,
    random_state=42,
    verbose=True,
)

# Train the model
print("Training the neural network on the entire dataset...")
model.fit(X_scaled, y)

# Evaluate the model
train_predictions = model.predict(X_scaled)
train_score = model.score(X_scaled, y)
train_mse = mean_squared_error(y, train_predictions)

print(f"\nTraining R² score: {train_score:.4f}")
print(f"Training MSE: {train_mse:.4f}")

# Save the model and scalers
os.makedirs("r", exist_ok=True)

print("\nSaving the model and scalers...")
joblib.dump(model, "r/racing_model.joblib")
joblib.dump(scaler, "r/scaler.joblib")
joblib.dump(steering_scaler, "r/steering_scaler.joblib")
print("Model and scalers saved successfully!")
