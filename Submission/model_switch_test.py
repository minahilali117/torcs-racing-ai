#!/usr/bin/env python
"""
Model switching test script for TORCS reinforcement learning.
This script manually tests the restart and model switching capabilities.
"""

import socket
import time
import sys


def send_meta_command(host="localhost", port=3001, restart=True):
    """Send meta command to TORCS to restart the race"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Create the meta command
    if restart:
        meta_cmd = "(meta 1)"
    else:
        meta_cmd = "(meta 0)"

    print(f"Sending command: {meta_cmd} to {host}:{port}")

    try:
        sock.sendto(meta_cmd.encode(), (host, port))
        print("Command sent successfully")
    except Exception as e:
        print(f"Error sending command: {e}")
    finally:
        sock.close()


def main():
    """Main function to test model switching"""
    if len(sys.argv) < 2:
        print("Usage: python model_switch_test.py <num_switches> [delay_seconds]")
        print("Example: python model_switch_test.py 3 30")
        sys.exit(1)

    num_switches = int(sys.argv[1])
    delay_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(
        f"Will perform {num_switches} model switches with {delay_seconds} seconds between each"
    )

    for i in range(num_switches):
        print(f"Switch {i+1}/{num_switches} - waiting {delay_seconds} seconds...")
        time.sleep(delay_seconds)
        send_meta_command()
        print("Restart command sent")

    print("Test complete")


if __name__ == "__main__":
    main()
