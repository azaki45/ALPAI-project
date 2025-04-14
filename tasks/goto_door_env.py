import gym
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class GoToDoorEnv(MiniGridEnv):
    """
    Custom MiniGrid environment where the agent must navigate toward a red door.
    The environment is defined on an 8x8 grid.
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
        obs, reward, done, info = super().step(action)
        # If the agent reaches the door, grant reward and finish the episode.
        if self.agent_pos == self.door_pos:
            reward = self._reward()
            done = True
        return obs, reward, done, info

# Register the environment with Gymnasium.
register(
    id='MiniGrid-GoToDoor-8x8-v0',
    entry_point='tasks.goto_door_env:GoToDoorEnv'
)
