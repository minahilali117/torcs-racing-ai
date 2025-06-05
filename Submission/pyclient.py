import sys
import argparse
import socket
import driver
import time
import os

if __name__ == "__main__":
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(
    description="Python client to connect to the TORCS SCRC server."
)

parser.add_argument(
    "--host",
    action="store",
    dest="host_ip",
    default="localhost",
    help="Host IP address (default: localhost)",
)
parser.add_argument(
    "--port",
    action="store",
    type=int,
    dest="host_port",
    default=3001,
    help="Host port number (default: 3001)",
)
parser.add_argument(
    "--id", action="store", dest="id", default="SCR", help="Bot ID (default: SCR)"
)
parser.add_argument(
    "--maxEpisodes",
    action="store",
    dest="max_episodes",
    type=int,
    default=100,
    help="Maximum number of learning episodes (default: 100)",
)
parser.add_argument(
    "--maxSteps",
    action="store",
    dest="max_steps",
    type=int,
    default=0,
    help="Maximum number of steps (default: 0)",
)
parser.add_argument(
    "--track", action="store", dest="track", default=None, help="Name of the track"
)
parser.add_argument(
    "--stage",
    action="store",
    dest="stage",
    type=int,
    default=3,
    help="Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)",
)

arguments = parser.parse_args()

# Print summary
print(
    "Connecting to server host ip:", arguments.host_ip, "@ port:", arguments.host_port
)
print("Bot ID:", arguments.id)
print("Maximum episodes:", arguments.max_episodes)
print("Maximum steps:", arguments.max_steps)
print("Track:", arguments.track)
print("Stage:", arguments.stage)
print("*********************************************")

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print("Could not make a socket.")
    sys.exit(-1)

# one second timeout
sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

verbose = False

# Initialize the driver
d = driver.Driver(arguments.stage)
d.training_mode = arguments.stage == 1  # Enable training in qualifying mode

# Ensure model directory exists
try:
    if not os.path.exists("model"):
        os.makedirs("model", exist_ok=True)
    if not os.path.exists("model/population"):
        os.makedirs("model/population", exist_ok=True)
    print("### VALIDATED MODEL DIRECTORIES ###")
except Exception as e:
    print(f"### ERROR CREATING MODEL DIRECTORIES: {e} ###")

# Variables for tracking model switches
current_model = 0
total_models = d.population_size if hasattr(d, "population_size") else 1

while not shutdownClient:
    while True:
        buf = arguments.id + d.init()

        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)

        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            # print("Didn't get response from server...")
            continue

        if buf and "***identified***" in buf:
            print("Received:", buf)
            break

    currentStep = 0
    print(f"Starting model {current_model + 1}/{total_models} evaluation")

    while True:
        # wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            # print("Didn't get response from server...")
            continue

        if buf and "***shutdown***" in buf:
            d.onShutDown()
            shutdownClient = True
            print("Client Shutdown")
            break

        if buf and "***restart***" in buf:
            print("### RESTART SIGNAL RECEIVED FROM TORCS ###")

            # When we've gone through all models, evolve population
            if (
                d.training_mode
                and d.current_model_idx == 0
                and hasattr(d, "evolve_population")
            ):
                print("### ALL MODELS EVALUATED, EVOLVING POPULATION ###")
                try:
                    # Ensure we evolve the population
                    success = d.evolve_population()
                    if success:
                        print(
                            f"### SUCCESSFULLY EVOLVED TO GENERATION {d.current_generation} ###"
                        )
                    else:
                        print("### EVOLUTION FAILED OR COMPLETED ###")
                except Exception as e:
                    print(f"### ERROR EVOLVING POPULATION: {e} ###")

            # Call driver's restart handler
            d.onRestart()
            print("### CLIENT RESTARTED SUCCESSFULLY ###")
            curEpisode += 1
            break

        # Process response and send control
        if buf:
            # Get driver's control response
            response = d.drive(buf)

            # Force model switch after 30 seconds
            if d.training_mode and time.time() - d.start_time > 30:
                print(
                    f"### 30-SECOND TIME LIMIT REACHED FOR MODEL {d.current_model_idx} ###"
                )

                # Calculate fitness
                fitness = d.calculate_fitness()
                d.fitness_scores[d.current_model_idx] = fitness
                print(f"### MODEL FITNESS: {fitness} ###")

                # DIRECTLY force the model to change - no waiting for restart
                next_model = (d.current_model_idx + 1) % d.population_size
                d.current_model_idx = next_model

                # Load the next model directly
                try:
                    d.model = d.models[d.current_model_idx]
                    d.scaler = d.scalers[d.current_model_idx]
                    d.steering_scaler = d.steering_scalers[d.current_model_idx]
                    print(f"### FORCED MODEL SWITCH TO MODEL {d.current_model_idx} ###")
                except Exception as e:
                    print(f"### ERROR SWITCHING MODEL: {e} ###")

                # Reset evaluation metrics
                d.reset_evaluation_metrics()
                d.start_time = time.time()

                # Force restart with direct TORCS command
                response = "(meta 1)"
                print("### SENDING RESTART COMMAND TO TORCS ###")

            # Only use driver's meta flag as a backup
            elif d.control.getMeta() == 1:
                print("### META FLAG SET, REQUESTING RESTART ###")
                response = "(meta 1)"
                d.control.setMeta(0)  # Reset the flag

            # Send the response
            try:
                sock.sendto(response.encode(), (arguments.host_ip, arguments.host_port))
                if verbose:
                    print(f"Sent: {response}")
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)

    if curEpisode == arguments.max_episodes:
        shutdownClient = True

sock.close()
