import gym
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class GoToDoorEnv(MiniGridEnv):
    """
    Custom MiniGrid environment where the agent must navigate toward a red door on an 8x8 grid.
    Reward shaping is applied:
      - At each step, the agent gets a small positive reward if it moves closer to the door.
      - When the agent reaches the door, a bonus reward is provided.
    """
    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            see_through_walls=True  # Useful for debugging.
        )
        self.door_pos = None

    def _gen_grid(self, width, height):
        # Create an empty grid with surrounding walls.
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Place a red door along the bottom wall at a random x-coordinate.
        door_x = self._rand_int(1, width - 1)
        self.grid.set(door_x, height - 1, Door("red", is_open=False))
        self.door_pos = (door_x, height - 1)
        
        # Place the agent somewhere in the upper half of the grid.
        self.agent_pos = (self._rand_int(1, width - 1), self._rand_int(1, height // 2))
        self.agent_dir = self._rand_int(0, 4)
        
        self.mission = "go to the red door"

    def step(self, action):
        # Compute the Manhattan distance from the agent to the door before taking the action.
        old_distance = abs(self.agent_pos[0] - self.door_pos[0]) + abs(self.agent_pos[1] - self.door_pos[1])
        
        # Take the action using the parent's step() method.
        obs, reward, done, info = super().step(action)
        
        # Compute the new Manhattan distance after the action.
        new_distance = abs(self.agent_pos[0] - self.door_pos[0]) + abs(self.agent_pos[1] - self.door_pos[1])
        
        # Shaping reward: reward is proportional to the reduction in distance (scaled by 0.1).
        shaping_reward = (old_distance - new_distance) * 0.1
        
        # If the agent reaches the door, assign a bonus reward.
        if self.agent_pos == self.door_pos:
            shaping_reward = 1.0  # Terminal bonus.
            done = True
        
        return obs, shaping_reward, done, info

# Register the environment using Gymnasium's registration.
register(
    id='MiniGrid-GoToDoor-8x8-v0',
    entry_point='tasks.goto_door_env:GoToDoorEnv'
)
