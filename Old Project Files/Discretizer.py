"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
"""

import gym
import numpy as np

class GalagaDiscretizer(gym.ActionWrapper):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super(GalagaDiscretizer, self).__init__(env)
        buttons = env.unwrapped.buttons
        actions = [["B"], ["LEFT"], ["LEFT", "B"], ["RIGHT"], ["RIGHT", "B"]]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):

    def reward(self, reward):
        return reward * 0.01


