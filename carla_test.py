# minimal_env_test.py
from embodied.envs.carla import Carla  # Import your wrapper
import traceback
import time
import numpy as np


def main():
    # Assume client/world are set up here, or inside your env's __init__
    env = Carla('task', repeat=2)

    for i in range(100):  # Test 10 consecutive resets
        print(f"--- Episode {i+1} ---")
        try:
            print("Calling env.reset()...")
            obs = env.env.reset()
            print("...env.reset() successful.")

            for step in range(50):  # Run a few steps
                print(f"  Step {step+1}")
                action = (np.random.rand(2) * 2) - 1
                acts = {'action': action, 'reset': False}
                obs = env.step(acts)
                if obs['is_last']:
                    print("  Episode finished early.")
                    break

        except Exception as e:
            print(f"!! An error occurred: {e}")
            print(traceback.format_exc())
            break

    env.close()
    print("Test finished.")


if __name__ == '__main__':
    main()
