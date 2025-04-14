import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class TrainLoggingCallback(BaseCallback):
    """
    Custom callback that logs training progress and saves training reward plots.
    """
    def __init__(self, log_freq=1000, save_path="./results", verbose=0):
        super(TrainLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.episodes = []
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check in the infos list if an episode has ended.
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if info.get("episode"):
                    r = info["episode"]["r"]
                    self.episode_rewards.append(r)
                    self.episodes.append(len(self.episodes) + 1)
                    if self.verbose:
                        print(f"Episode {len(self.episodes)} reward: {r}")
                    
                    # Save a plot every log_freq episodes
                    if len(self.episodes) % self.log_freq == 0:
                        self.plot_rewards()
        return True

    def plot_rewards(self):
        if len(self.episode_rewards) == 0:
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
        # Final reward plot at end of training.
        self.plot_rewards()
