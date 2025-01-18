import time

import gym


def main():
    # Create the Pendulum-v1 environment
    env = gym.make("Pendulum-v1")

    # Reset the environment to get the initial observation
    observation = env.reset()

    try:
        while True:
            # Render the environment
            env.render()

            # Sample a random action from the action space
            action = env.action_space.sample()

            # Take a step in the environment using the sampled action
            observation, reward, done, info = env.step(action)

            # Optional: Slow down the loop for better visualization
            time.sleep(0.02)

            # If the episode ends (not typical for pendulum), reset
            if done:
                _ = env.reset()
    except KeyboardInterrupt:
        # Close the environment when the user interrupts
        print("Exiting gracefully...")
    finally:
        env.close()


if __name__ == "__main__":
    main()
