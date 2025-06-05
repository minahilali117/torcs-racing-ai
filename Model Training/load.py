import joblib
import pandas as pd
import numpy as np

# Load model and scalers
model = joblib.load("racing_model.joblib")
scaler = joblib.load("scaler.joblib")
steering_scaler = joblib.load("steering_scaler.joblib")

# Define the input columns expected by the model
input_columns = [
    "Track_1",
    "Track_5",
    "Track_10",
    "Track_15",
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

# Your data row (must match input_columns in order and length)
data_row = [
    6.66667,  # Track_1
    19.492,  # Track_5
    85.9487,  # Track_10
    38.2456,  # Track_15
    4.35136,  # Track_19
    94.0,  # SpeedX
    0.0,  # SpeedY
    0.0,  # SpeedZ
    1.74846e-7,  # Angle
    10.5409,  # TrackPosition
    942.478,  # RPM
    0.345258,  # WheelSpinVelocity_1
    1.0,  # WheelSpinVelocity_2
    0.0,  # WheelSpinVelocity_3
    0.63999999,  # WheelSpinVelocity_4
    0.0,  # DistanceCovered
    2818.1,  # DistanceFromStart
    -0.982,  # CurrentLapTime
    0.0,  # Damage
    200.0,  # Opponent_9
    200.0,  # Opponent_10
    200.0,  # Opponent_11
    200.0,  # Opponent_19
]

# Convert to DataFrame
X_new = pd.DataFrame([data_row], columns=input_columns)

# Scale the data
X_new_scaled = scaler.transform(X_new)

# Predict
output = model.predict(X_new_scaled)

# Convert the prediction to a DataFrame for easier manipulation
output_df = pd.DataFrame(output, columns=["Steering", "Acceleration", "Braking"])

# Inverse-transform only Steering
output_df[["Steering"]] = steering_scaler.inverse_transform(output_df[["Steering"]])
output_df["Steering"] = np.clip(output_df["Steering"], -1, 1)

# Post-process the output to match the required format
output_df["Acceleration"] = (output_df["Acceleration"] > 0.5).astype(int)
output_df["Braking"] = (output_df["Braking"] > 0.5).astype(int)

print("Predicted output:")
print(output_df)
