# minimal_env_test.py
from embodied.envs.carla import Carla  # Import your wrapper
import time


def main():
    # Assume client/world are set up here, or inside your env's __init__
    env = Carla('task', repeat=2)

    for i in range(10):  # Test 10 consecutive resets
        print(f"--- Episode {i+1} ---")
        try:
            print("Calling env.reset()...")
            obs = env.reset()
            print("...env.reset() successful.")

            for step in range(50):  # Run a few steps
                print(f"  Step {step+1}")
                action = env.action_space.sample()  # Use a random action
                obs, reward, done, info = env.step(action)
                if done:
                    print("  Episode finished early.")
                    break
                time.sleep(0.1)  # Slow it down to see prints

        except Exception as e:
            print(f"!! An error occurred: {e}")
            break

    env.close()
    print("Test finished.")


if __name__ == '__main__':
    main()
