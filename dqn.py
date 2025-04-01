import minigrid
import cv2
import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper
from extractor import MinigridFeaturesExtractor
from stable_baselines3 import PPO,DQN

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = gym.make("MiniGrid-DistShift2-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)

model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(1e5)


episodes = 10
vec_env = model.get_env()

for ep in range(episodes):
	obs = vec_env.reset()
	done = False
	while not done:
		action, _states = model.predict(obs)
		obs, rewards, done, info = vec_env.step(action)
		vec_env.render("human")
		print(rewards)
        
vec_env.close()

