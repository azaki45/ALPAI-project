import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from minigrid.wrappers import ImgObsWrapper
from extractor import MinigridFeaturesExtractor
from callbacks import TrainLoggingCallback

# Define policy customization using the custom feature extractor.
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# Create the GoToDoor environment.
env = gym.make("MiniGrid-GoToDoor-8x8-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

# Initialize the DQN model with the CNN-based policy.
model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# Initialize the training logging callback.
callback = TrainLoggingCallback(log_freq=20, save_path="./results", verbose=1)

# Train the model (adjust timesteps as needed).
model.learn(total_timesteps=50000, callback=callback)

# Save the trained model.
os.makedirs("./results", exist_ok=True)
model.save("./results/dqn_gotodoor_model")

# -------------------------------
# Evaluation: Run several episodes to collect performance data.
# -------------------------------
episodes = 10
results = {"episode": [], "total_reward": [], "steps": []}

for ep in range(episodes):
    # Unpack observation and info from reset()
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    while not done:
        action, _ = model.predict(obs)
        # Unpack five values: obs, reward, terminated, truncated, info.
        obs, reward, terminated, truncated, info = env.step(action)
        # Determine end of episode.
        done = terminated or truncated
        # Call render with no additional parameter.
        env.render()  
        episode_reward += reward
        steps += 1
    results["episode"].append(ep + 1)
    results["total_reward"].append(episode_reward)
    results["steps"].append(steps)
    print(f"Episode {ep+1}: Total Reward = {episode_reward}, Steps = {steps}")

env.close()

# Save evaluation results to a CSV file.
df = pd.DataFrame(results)
df.to_csv("./results/evaluation_results.csv", index=False)
print("Evaluation results saved to './results/evaluation_results.csv'")
