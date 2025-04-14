import os
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class TrainLoggingCallback(BaseCallback):
    """
    Custom callback that logs training progress, saves reward plots,
    and writes each training episode's data (episode number, reward, length)
    to a CSV file.
    """
    def __init__(self, log_freq=20, save_path="./results", verbose=0):
        super(TrainLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.episodes = []
        self.training_data = []  # To store each episode's data
        os.makedirs(self.save_path, exist_ok=True)
        self.csv_file_path = os.path.join(self.save_path, "training_episodes.csv")

    def _on_step(self) -> bool:
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if info.get("episode"):
                    ep_info = info["episode"]
                    # Retrieve reward and length if available
                    reward = ep_info.get("r", 0)
                    length = ep_info.get("l", None)
                    episode_number = len(self.episodes) + 1

                    self.episode_rewards.append(reward)
                    self.episodes.append(episode_number)
                    self.training_data.append({
                        "episode": episode_number,
                        "reward": reward,
                        "length": length
                    })

                    if self.verbose:
                        print(f"Episode {episode_number} reward: {reward}")

                    # Append current episode info to CSV
                    file_exists = os.path.isfile(self.csv_file_path)
                    with open(self.csv_file_path, "a") as f:
                        if not file_exists:
                            f.write("episode,reward,length\n")
                        f.write(f"{episode_number},{reward},{length if length is not None else ''}\n")

                    # Save a plot every log_freq episodes.
                    if episode_number % self.log_freq == 0:
                        self.plot_rewards()
        return True

    def plot_rewards(self):
        if not self.episode_rewards:
            return
        plt.figure()
        plt.plot(self.episodes, self.episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Reward Curve")
        plt.legend()
        plot_file = os.path.join(self.save_path, f"training_rewards_{len(self.episodes)}eps.png")
        plt.savefig(plot_file)
        plt.close()
        if self.verbose:
            print(f"Saved reward plot to {plot_file}")

    def _on_training_end(self) -> None:
        # Final plot at the end of training.
        self.plot_rewards()
        # Optionally, overwrite the CSV file with complete data for consistency.
        df = pd.DataFrame(self.training_data)
        df.to_csv(self.csv_file_path, index=False)
        if self.verbose:
            print(f"Training episode data saved to {self.csv_file_path}")
