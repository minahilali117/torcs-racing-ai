import msgParser
import carState
import carControl
import joblib
import pandas as pd
import numpy as np
import copy
import time
import os
import random
import glob
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class Driver(object):
    """
    A driver object for the SCRC
    """

    def __init__(self, stage):
        """Constructor"""
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage

        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None

        # Track the current gear internally since getGear() returns None
        self.current_gear = 1

        # Add gear shifting cooldown to prevent oscillation
        self.last_gear_change_time = 0
        self.gear_shift_cooldown = 1.0  # 1 second cooldown between shifts

        # Track position correction parameters
        self.track_position_correction_factor = (
            0.3  # Adjustable parameter for steering correction
        )

        # Acceleration burst parameters
        self.accel_burst_active = False
        self.accel_burst_start_time = 0
        self.accel_burst_duration = 0.5  # Duration of acceleration burst in seconds

        # Reverse mode parameters
        self.reverse_mode_active = False
        self.race_started = False
        self.race_start_time = 0
        self.stuck_threshold_speed = 5.0  # Speed below which we consider the car stuck
        self.stuck_threshold_time = 2.0  # Time to consider car as stuck (seconds)
        self.stuck_accel_threshold = (
            0.8  # High acceleration threshold to consider "max accel"
        )
        self.stuck_start_time = 0
        self.angle_threshold = 0.1  # Threshold for exiting reverse mode
        self.reverse_steer_factor = 0.8  # Steering factor for reverse
        self.reverse_mode_duration = 2.0  # Fixed duration for reverse mode in seconds
        self.reverse_start_time = 0  # When reverse mode started

        # Stuck in reverse detection
        self.reverse_stuck_start_time = 0  # When the car became stuck in reverse
        self.reverse_stuck_threshold_speed = (
            0.3  # Speed below which we consider the car stuck in reverse
        )
        self.reverse_stuck_threshold_time = (
            1.5  # Time to consider car as stuck in reverse (seconds)
        )

        # Simple model parameters
        self.model = None
        self.scaler = None
        self.steering_scaler = None

        # Load the model
        self.load_model()

    def load_model(self):
        """Load the racing model and scalers"""
        print("Loading model...")

        # Make sure model directory exists
        os.makedirs("model", exist_ok=True)

        try:
            # Load model
            model_path = "model/racing_model.joblib"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"Loaded model from {model_path}")
            else:
                print(f"WARNING: Model not found at {model_path}")
                # Create a dummy model if no model exists
                self.model = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=1000)
                # Dummy train to initialize
                dummy_X = np.random.rand(10, 23)
                dummy_y = np.random.rand(10, 3)
                self.model.fit(dummy_X, dummy_y)
                print("Created new dummy model")
                # Save it
                joblib.dump(self.model, model_path)

            # Load or create scaler
            scaler_path = "model/scaler.joblib"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("Loaded scaler")
            else:
                self.scaler = StandardScaler()
                # Dummy fit
                self.scaler.fit(np.random.rand(10, 23))
                joblib.dump(self.scaler, scaler_path)
                print("Created new scaler")

            # Load or create steering scaler
            steering_scaler_path = "model/steering_scaler.joblib"
            if os.path.exists(steering_scaler_path):
                self.steering_scaler = joblib.load(steering_scaler_path)
                print("Loaded steering scaler")
            else:
                self.steering_scaler = StandardScaler()
                # Dummy fit
                self.steering_scaler.fit(np.random.rand(10, 1))
                joblib.dump(self.steering_scaler, steering_scaler_path)
                print("Created new steering scaler")

            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create emergency dummy model
            self.model = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=1000)
            dummy_X = np.random.rand(10, 23)
            dummy_y = np.random.rand(10, 3)
            self.model.fit(dummy_X, dummy_y)
            self.scaler = StandardScaler().fit(dummy_X)
            self.steering_scaler = StandardScaler().fit(dummy_y[:, 0].reshape(-1, 1))
            print("Created emergency dummy model due to error")

    def init(self):
        """Return init string with rangefinder angles"""
        self.angles = [0 for x in range(19)]

        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15

        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5

        return self.parser.stringify({"init": self.angles})

    def prevent_simultaneous_accel_brake(self):
        """Prevent acceleration and brake from being active at the same time"""
        accel = self.control.getAccel()
        brake = self.control.getBrake()

        # If both are active, prioritize braking
        if accel > 0.1 and brake > 0.1:
            print(
                f"SAFETY: Preventing simultaneous accel ({accel:.2f}) and brake ({brake:.2f})"
            )
            # Keep the brake, zero out acceleration
            self.control.setAccel(0)

    def predict_and_set_controls(self):
        """Main control function that uses the model to control the car"""
        # Input features for the model
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

        # Build input row from current state
        data_row = [
            # Track sensors 1-19
            (
                self.state.track[0]
                if self.state.track and len(self.state.track) > 0
                else 0.0
            ),
            (
                self.state.track[1]
                if self.state.track and len(self.state.track) > 1
                else 0.0
            ),
            (
                self.state.track[2]
                if self.state.track and len(self.state.track) > 2
                else 0.0
            ),
            (
                self.state.track[3]
                if self.state.track and len(self.state.track) > 3
                else 0.0
            ),
            (
                self.state.track[4]
                if self.state.track and len(self.state.track) > 4
                else 0.0
            ),
            (
                self.state.track[5]
                if self.state.track and len(self.state.track) > 5
                else 0.0
            ),
            (
                self.state.track[6]
                if self.state.track and len(self.state.track) > 6
                else 0.0
            ),
            (
                self.state.track[7]
                if self.state.track and len(self.state.track) > 7
                else 0.0
            ),
            (
                self.state.track[8]
                if self.state.track and len(self.state.track) > 8
                else 0.0
            ),
            (
                self.state.track[9]
                if self.state.track and len(self.state.track) > 9
                else 0.0
            ),
            (
                self.state.track[10]
                if self.state.track and len(self.state.track) > 10
                else 0.0
            ),
            (
                self.state.track[11]
                if self.state.track and len(self.state.track) > 11
                else 0.0
            ),
            (
                self.state.track[12]
                if self.state.track and len(self.state.track) > 12
                else 0.0
            ),
            (
                self.state.track[13]
                if self.state.track and len(self.state.track) > 13
                else 0.0
            ),
            (
                self.state.track[14]
                if self.state.track and len(self.state.track) > 14
                else 0.0
            ),
            (
                self.state.track[15]
                if self.state.track and len(self.state.track) > 15
                else 0.0
            ),
            (
                self.state.track[16]
                if self.state.track and len(self.state.track) > 16
                else 0.0
            ),
            (
                self.state.track[17]
                if self.state.track and len(self.state.track) > 17
                else 0.0
            ),
            (
                self.state.track[18]
                if self.state.track and len(self.state.track) > 18
                else 0.0
            ),
            self.state.speedX if self.state.speedX is not None else 0.0,
            self.state.speedY if self.state.speedY is not None else 0.0,
            self.state.speedZ if self.state.speedZ is not None else 0.0,
            self.state.angle if self.state.angle is not None else 0.0,
            self.state.trackPos if self.state.trackPos is not None else 0.0,
            self.state.rpm if self.state.rpm is not None else 0.0,
            (
                self.state.wheelSpinVel[0]
                if self.state.wheelSpinVel and len(self.state.wheelSpinVel) > 0
                else 0.0
            ),
            (
                self.state.wheelSpinVel[1]
                if self.state.wheelSpinVel and len(self.state.wheelSpinVel) > 1
                else 0.0
            ),
            (
                self.state.wheelSpinVel[2]
                if self.state.wheelSpinVel and len(self.state.wheelSpinVel) > 2
                else 0.0
            ),
            (
                self.state.wheelSpinVel[3]
                if self.state.wheelSpinVel and len(self.state.wheelSpinVel) > 3
                else 0.0
            ),
            self.state.distRaced if self.state.distRaced is not None else 0.0,
            self.state.distFromStart if self.state.distFromStart is not None else 0.0,
            self.state.curLapTime if self.state.curLapTime is not None else 0.0,
            self.state.damage if self.state.damage is not None else 0.0,
            (
                self.state.opponents[8]
                if self.state.opponents and len(self.state.opponents) > 8
                else 0.0
            ),
            (
                self.state.opponents[9]
                if self.state.opponents and len(self.state.opponents) > 9
                else 0.0
            ),
            (
                self.state.opponents[10]
                if self.state.opponents and len(self.state.opponents) > 10
                else 0.0
            ),
            (
                self.state.opponents[18]
                if self.state.opponents and len(self.state.opponents) > 18
                else 0.0
            ),
        ]

        # Create DataFrame from input row
        X_new = pd.DataFrame([data_row], columns=input_columns)

        # Scale input data
        X_new_scaled = self.scaler.transform(X_new)

        # Get model prediction
        output = self.model.predict(X_new_scaled)

        # Create DataFrame for output
        output_df = pd.DataFrame(
            output, columns=["Steering", "Acceleration", "Braking"]
        )

        # Scale steering output
        output_df[["Steering"]] = self.steering_scaler.inverse_transform(
            output_df[["Steering"]]
        )

        # Process outputs
        output_df["Steering"] = np.clip(output_df["Steering"], -1, 1)
        output_df["Acceleration"] = (output_df["Acceleration"] > 0.5).astype(int)
        output_df["Braking"] = (output_df["Braking"] > 0.5).astype(int)

        # Get the model's steering output
        model_steering = float(output_df["Steering"].iloc[0])

        # Apply track position correction
        track_pos = self.state.trackPos if self.state.trackPos is not None else 0

        # Apply correction for both sides of the track:
        # - If car is to the right (positive track_pos), add left steering (negative value)
        # - If car is to the left (negative track_pos), add right steering (positive value)
        # The further from center, the stronger the correction
        correction = -track_pos * self.track_position_correction_factor
        corrected_steering = model_steering + correction

        # # Only print debug if a significant correction was applied
        # if abs(correction) > 0.01:
        #     print(
        #         f"TRACK CORRECTION: Position {track_pos:.2f}, adding {correction:.2f} to steering. "
        #         f"Original: {model_steering:.2f}, Corrected: {corrected_steering:.2f}"
        #     )

        # Clip the steering to ensure it stays within valid range
        corrected_steering = np.clip(corrected_steering, -1, 1)

        # Get current speed and acceleration
        speed = abs(self.state.speedX) if self.state.speedX is not None else 0
        current_time = self.state.curLapTime if self.state.curLapTime is not None else 0
        accel = float(output_df["Acceleration"].iloc[0])

        # Check if race has started (we've been running for a few seconds)
        if current_time > 5.0 and not self.race_started:
            self.race_started = True
            self.race_start_time = current_time
            print(f"RACE STARTED at time {current_time:.2f}")

        # Handle reverse mode logic
        if self.reverse_mode_active:
            # Check how long we've been in reverse mode
            time_in_reverse = current_time - self.reverse_start_time

            # Get current speed (absolute value for simplicity)
            speed = abs(self.state.speedX) if self.state.speedX is not None else 0

            # Check if the car is stuck in reverse (very low speed despite being in reverse with acceleration)
            if speed < self.reverse_stuck_threshold_speed:
                # Car might be stuck in reverse
                if self.reverse_stuck_start_time == 0:
                    # Start tracking stuck in reverse time
                    self.reverse_stuck_start_time = current_time
                    print(
                        f"CAR MIGHT BE STUCK IN REVERSE: Speed {speed:.2f} < {self.reverse_stuck_threshold_speed}"
                    )
                elif (
                    current_time - self.reverse_stuck_start_time
                    > self.reverse_stuck_threshold_time
                ):
                    # Car has been stuck in reverse for too long - exit reverse mode
                    self.reverse_mode_active = False
                    self.reverse_stuck_start_time = 0  # Reset timer
                    print(
                        f"EXITING REVERSE MODE: Car stuck in reverse for {self.reverse_stuck_threshold_time:.1f} seconds"
                    )

                    # Reset to normal gear
                    self.current_gear = 1
                    self.control.setGear(1)
                    return  # Continue with normal control
            else:
                # Car is moving in reverse - reset stuck timer
                self.reverse_stuck_start_time = 0

            # Check if we've been in reverse mode long enough (fixed 2-second duration)
            if time_in_reverse >= self.reverse_mode_duration:
                self.reverse_mode_active = False
                print(
                    f"EXITING REVERSE MODE: Completed fixed {self.reverse_mode_duration}s reverse duration"
                )

                # Reset to normal gear when exiting reverse
                self.current_gear = 1
                self.control.setGear(1)
            else:
                # Still in reverse mode - set control values for proper reverse

                # Set to reverse gear (-1)
                self.current_gear = -1
                self.control.setGear(-1)

                # Use acceleration (not brake) to move backward in reverse gear
                self.control.setAccel(0.8)  # Strong acceleration backward
                self.control.setBrake(0)  # No brake

                # Safety check
                self.prevent_simultaneous_accel_brake()

                # Steer based on current angle
                angle_value = self.state.angle if self.state.angle is not None else 0
                # If angle is positive, steer right (positive value) to straighten out
                # If angle is negative, steer left (negative value) to straighten out
                reverse_steer = -np.sign(angle_value) * self.reverse_steer_factor
                self.control.setSteer(reverse_steer)

                # Show more detailed debug info including time in reverse
                print(
                    f"REVERSE MODE: {time_in_reverse:.1f}s/{self.reverse_mode_duration:.1f}s, Angle {angle_value:.4f}, Gear {self.current_gear}, Steer {reverse_steer:.4f}"
                )

                # Skip the rest of the control setting
                return self.control.toMsg()
        else:
            # Not in reverse mode - check if we should enter it
            # Conditions:
            # 1. Race has started
            # 2. Speed is below threshold (< 5)
            # 3. Car is at high acceleration (> 0.8) - trying hard to move forward
            if (
                self.race_started
                and speed < self.stuck_threshold_speed
                and accel > self.stuck_accel_threshold
            ):
                # Car might be stuck - check if it's been stuck for the threshold time
                if self.stuck_start_time == 0:
                    # Start tracking stuck time
                    self.stuck_start_time = current_time
                    print(
                        f"CAR MIGHT BE STUCK: Speed {speed:.2f} < {self.stuck_threshold_speed}, Accel {accel:.2f} > {self.stuck_accel_threshold}"
                    )
                elif current_time - self.stuck_start_time > self.stuck_threshold_time:
                    # Car has been stuck for long enough - activate reverse mode
                    self.reverse_mode_active = True
                    self.stuck_start_time = 0  # Reset stuck timer
                    self.reverse_start_time = (
                        current_time  # Record when reverse mode started
                    )
                    print(
                        f"ACTIVATING REVERSE MODE: Slow speed with high accel for {self.stuck_threshold_time:.1f} seconds"
                    )

                    # Set initial reverse controls
                    self.current_gear = -1  # Set to reverse gear
                    self.control.setGear(-1)
                    self.control.setAccel(0.8)  # Strong acceleration for reverse
                    self.control.setBrake(0)  # No brake

                    # Safety check
                    self.prevent_simultaneous_accel_brake()

                    # Initial reverse steering based on current angle
                    angle_value = (
                        self.state.angle if self.state.angle is not None else 0
                    )
                    reverse_steer = -np.sign(angle_value) * self.reverse_steer_factor
                    self.control.setSteer(reverse_steer)

                    print(
                        f"INITIAL REVERSE: Setting gear {self.current_gear}, steer {reverse_steer:.4f}, accel 0.8"
                    )

                    # Skip the rest of the control setting
                    return self.control.toMsg()
            else:
                # Car is moving or no acceleration - reset stuck timer
                self.stuck_start_time = 0

        # Set the control values (if not in reverse mode)
        self.control.setSteer(corrected_steering)

        # Get wheel spin velocities for traction control
        wheel_spins = []
        if self.state.wheelSpinVel is not None and len(self.state.wheelSpinVel) == 4:
            wheel_spins = self.state.wheelSpinVel

            # Calculate wheel spin differences
            # Consider paired wheels (front pair and rear pair)
            front_diff = abs(wheel_spins[0] - wheel_spins[1])
            rear_diff = abs(wheel_spins[2] - wheel_spins[3])

            # Threshold for wheel spin difference - lowered to be more sensitive
            wheel_spin_threshold = 5.0  # Lower threshold to trigger more easily

            # Debug - always print wheel spin values to diagnose
            print(
                f"WHEEL SPINS: F1:{wheel_spins[0]:.2f}, F2:{wheel_spins[1]:.2f}, R1:{wheel_spins[2]:.2f}, R2:{wheel_spins[3]:.2f}"
            )
            print(
                f"WHEEL DIFFS: Front:{front_diff:.2f}, Rear:{rear_diff:.2f}, Threshold:{wheel_spin_threshold}"
            )

            # Check if any pair has significant difference (wheel slip)
            if front_diff > wheel_spin_threshold or rear_diff > wheel_spin_threshold:
                # Wheel spin detected - reduce acceleration to a quarter
                model_accel = float(output_df["Acceleration"].iloc[0])
                reduced_accel = 0.1  # Force very low fixed value to ensure it's working
                self.control.setAccel(reduced_accel)
                print(
                    f"TRACTION CONTROL ACTIVE! Wheel spin detected (F:{front_diff:.2f}, R:{rear_diff:.2f}) - Reducing accel from {model_accel:.2f} to {reduced_accel:.2f}"
                )
            else:
                # Normal acceleration
                model_accel = float(output_df["Acceleration"].iloc[0])
                self.control.setAccel(model_accel)
                print(f"NORMAL TRACTION: Using full accel {model_accel:.2f}")
        else:
            # No wheel data - use normal acceleration
            print("WARNING: No wheel spin data available")

        self.control.setBrake(float(output_df["Braking"].iloc[0]))

        # Safety check to prevent simultaneous accel and brake
        self.prevent_simultaneous_accel_brake()

        # Check if car is stuck (not moving and no accel/brake commands)
        speed = abs(self.state.speedX) if self.state.speedX is not None else 0
        accel = self.control.getAccel()
        brake = self.control.getBrake()
        current_time = self.state.curLapTime if self.state.curLapTime is not None else 0

        # Handle acceleration bursts with duration
        if self.accel_burst_active:
            # Check if burst duration has elapsed
            if current_time - self.accel_burst_start_time >= self.accel_burst_duration:
                # End the burst
                self.accel_burst_active = False
                print(
                    f"ACCEL BURST: Ended after {current_time - self.accel_burst_start_time:.2f}s"
                )
            else:
                # Continue the acceleration burst
                self.control.setAccel(0.8)
                self.control.setBrake(0)
                print(
                    f"ACCEL BURST: Continuing, elapsed {current_time - self.accel_burst_start_time:.2f}s / {self.accel_burst_duration:.2f}s"
                )
        else:
            # If speed is very low, no acceleration and no brake, start a new acceleration burst
            if speed < 0.5 and accel < 0.1 and brake < 0.1:
                self.accel_burst_active = True
                self.accel_burst_start_time = current_time
                self.control.setAccel(
                    0.8
                )  # Apply a significant but not full acceleration
                self.control.setBrake(0)  # Ensure brake is off
                print(
                    f"ACCEL BURST: Starting new burst at time {current_time:.2f}s, duration {self.accel_burst_duration:.2f}s"
                )

        # Check safety again after potential updates
        self.prevent_simultaneous_accel_brake()

        # Set gear based on RPM and speed
        self.set_gear()

    def set_gear(self):
        """Set gear based on RPM and speed"""
        # Get current gear from our internal tracking, RPM and speed
        rpm = self.state.rpm if self.state.rpm is not None else 0
        speed = abs(self.state.speedX) if self.state.speedX is not None else 0
        current_time = self.state.curLapTime if self.state.curLapTime is not None else 0

        # Get acceleration and brake values directly from control
        accel = self.control.getAccel() if hasattr(self.control, "getAccel") else 0
        brake = self.control.getBrake() if hasattr(self.control, "getBrake") else 0

        # Check if we're in reverse mode
        if self.reverse_mode_active:
            self.current_gear = -1
            self.control.setGear(-1)
            return

        # Simple gear shifting up to second gear only
        # Define RPM thresholds for gear changes
        upshift_threshold = 8000  # RPM threshold for upshifting
        downshift_threshold = 3000  # RPM threshold for downshifting

        # Gear shift cooldown to prevent oscillation
        if current_time - self.last_gear_change_time < self.gear_shift_cooldown:
            # Still in cooldown period - don't change gears
            self.control.setGear(self.current_gear)
            return

        # Upshift logic - if RPM is high and we're in first gear
        if rpm > upshift_threshold and self.current_gear == 1:
            # Min speed check before upshifting to 2nd gear
            if speed > 25:  # Minimum speed for 2nd gear
                self.current_gear = 2
                self.last_gear_change_time = current_time
                print(f"GEAR: Upshifting to 2nd at RPM {rpm:.0f}")

        # Downshift logic - if RPM is low and we're in second gear
        elif rpm < downshift_threshold and self.current_gear == 2:
            self.current_gear = 1
            self.last_gear_change_time = current_time
            print(f"GEAR: Downshifting to 1st at RPM {rpm:.0f}")

        # Special case - if car is almost stopped, go to first gear
        if speed < 5 and self.current_gear != 1 and not brake > 0.5:
            self.current_gear = 1
            self.last_gear_change_time = current_time
            print("GEAR: Resetting to first gear due to low speed")

        # Apply the gear change
        self.control.setGear(self.current_gear)

    def get_input_dataframe(self):
        """Get current state as a DataFrame"""
        data = {
            "Track_1": (
                self.state.track[0]
                if self.state.track and len(self.state.track) > 0
                else None
            ),
            "Track_2": (
                self.state.track[1]
                if self.state.track and len(self.state.track) > 1
                else None
            ),
            "Track_3": (
                self.state.track[2]
                if self.state.track and len(self.state.track) > 2
                else None
            ),
            "Track_4": (
                self.state.track[3]
                if self.state.track and len(self.state.track) > 3
                else None
            ),
            "Track_5": (
                self.state.track[4]
                if self.state.track and len(self.state.track) > 4
                else None
            ),
            "Track_6": (
                self.state.track[5]
                if self.state.track and len(self.state.track) > 5
                else None
            ),
            "Track_7": (
                self.state.track[6]
                if self.state.track and len(self.state.track) > 6
                else None
            ),
            "Track_8": (
                self.state.track[7]
                if self.state.track and len(self.state.track) > 7
                else None
            ),
            "Track_9": (
                self.state.track[8]
                if self.state.track and len(self.state.track) > 8
                else None
            ),
            "Track_10": (
                self.state.track[9]
                if self.state.track and len(self.state.track) > 9
                else None
            ),
            "Track_11": (
                self.state.track[10]
                if self.state.track and len(self.state.track) > 10
                else None
            ),
            "Track_12": (
                self.state.track[11]
                if self.state.track and len(self.state.track) > 11
                else None
            ),
            "Track_13": (
                self.state.track[12]
                if self.state.track and len(self.state.track) > 12
                else None
            ),
            "Track_14": (
                self.state.track[13]
                if self.state.track and len(self.state.track) > 13
                else None
            ),
            "Track_15": (
                self.state.track[14]
                if self.state.track and len(self.state.track) > 14
                else None
            ),
            "Track_16": (
                self.state.track[15]
                if self.state.track and len(self.state.track) > 15
                else None
            ),
            "Track_17": (
                self.state.track[16]
                if self.state.track and len(self.state.track) > 16
                else None
            ),
            "Track_18": (
                self.state.track[17]
                if self.state.track and len(self.state.track) > 17
                else None
            ),
            "Track_19": (
                self.state.track[18]
                if self.state.track and len(self.state.track) > 18
                else None
            ),
            "SpeedX": self.state.speedX,
            "SpeedY": self.state.speedY,
            "SpeedZ": self.state.speedZ,
            "Angle": self.state.angle,
            "TrackPosition": self.state.trackPos,
            "RPM": self.state.rpm,
            "WheelSpinVelocity_1": (
                self.state.wheelSpinVel[0]
                if self.state.wheelSpinVel and len(self.state.wheelSpinVel) > 0
                else None
            ),
            "WheelSpinVelocity_2": (
                self.state.wheelSpinVel[1]
                if self.state.wheelSpinVel and len(self.state.wheelSpinVel) > 1
                else None
            ),
            "WheelSpinVelocity_3": (
                self.state.wheelSpinVel[2]
                if self.state.wheelSpinVel and len(self.state.wheelSpinVel) > 2
                else None
            ),
            "WheelSpinVelocity_4": (
                self.state.wheelSpinVel[3]
                if self.state.wheelSpinVel and len(self.state.wheelSpinVel) > 3
                else None
            ),
            "DistanceCovered": self.state.distRaced,
            "DistanceFromStart": self.state.distFromStart,
            "CurrentLapTime": self.state.curLapTime,
            "Damage": self.state.damage,
            "Opponent_9": (
                self.state.opponents[8]
                if self.state.opponents and len(self.state.opponents) > 8
                else None
            ),
            "Opponent_10": (
                self.state.opponents[9]
                if self.state.opponents and len(self.state.opponents) > 9
                else None
            ),
            "Opponent_11": (
                self.state.opponents[10]
                if self.state.opponents and len(self.state.opponents) > 10
                else None
            ),
            "Opponent_19": (
                self.state.opponents[18]
                if self.state.opponents and len(self.state.opponents) > 18
                else None
            ),
        }
        return pd.DataFrame([data])

    def drive(self, msg):
        """Process message from TORCS and generate control commands"""
        self.state.setFromMsg(msg)
        self.predict_and_set_controls()

        # Final safety check before sending controls
        self.prevent_simultaneous_accel_brake()

        return self.control.toMsg()

    def onShutDown(self):
        """Called when the race is over"""
        print("Driver shutdown")
        self.state.end_game()

    def onRestart(self):
        """Called when the race server restarts the race"""
        print("Driver restarted")
