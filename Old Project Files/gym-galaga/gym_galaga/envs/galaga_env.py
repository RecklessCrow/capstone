import subprocess
from time import sleep
import gym
import pyautogui
import Computer_Vision

# Constants
# You may need to change your control scheme in Mesen to fit these inputs
RESET0  = 'ctrl'
RESET1  = 'r'
START   = 'e'
LEFT    = 'a'
RIGHT   = 'd'
FIRE    = 'j'


class GalagaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # TODO launch game and focus window
        # subprocess.call(["bash", "launch_headless.bash"])

        # Get window information
        window_pos_and_size = subprocess.check_output(["bash", "window_location_script.bash"])
        window_pos_and_size = window_pos_and_size.split(b', ')
        menu_offset = 30  # Toolbar of Mesen just looked unattractive, use this to get rid of it
        width = int(window_pos_and_size[0])
        height = int(window_pos_and_size[1]) - menu_offset
        x = int(window_pos_and_size[2])
        y = int(window_pos_and_size[3]) + menu_offset

        self.window = {
            "top": y,
            "left": x,
            "width": width,
            "height": height
        }

        self.old_score = 0
        self.reward_multiplier = 1000
        # self.start_time = 0

    def step(self, action):

        # Possible actions of the model
        if action == 0:
            pyautogui.keyUp(RIGHT)
            pyautogui.keyUp(LEFT)
        elif action == 1:
            pyautogui.keyDown(LEFT)
            pyautogui.keyUp(LEFT)
        elif action == 2:
            pyautogui.keyDown(RIGHT)
            pyautogui.keyUp(RIGHT)
        elif action == 3:
            pyautogui.keyDown(LEFT)
        elif action == 4:
            pyautogui.keyDown(RIGHT)

        # Get observation
        obs, score, msg = Computer_Vision.observe(self.window)

        partial_reward = score - self.old_score

        reward = 0

        if   partial_reward >= 3000:
            reward = 1.0
        elif partial_reward >= 1600:
            reward = 0.5
        elif partial_reward >= 800:
            reward = 0.4
        elif partial_reward >= 400:
            reward = 0.3
        elif partial_reward >= 200:
            reward = 0.2
        elif partial_reward >= 100:
            reward = 0.1

        self.old_score = score

        done = False

        if msg.lower() == 'ready':
            reward = -0.5

        if msg.lower() == 'game over':
            reward = -15
            done = True

        info = {}

        return obs, reward, done, info

    def reset(self):

        # self.start_time = time()

        # If some keys were still being pressed unpress them
        pyautogui.keyUp(RIGHT)
        pyautogui.keyUp(LEFT)
        pyautogui.keyUp(FIRE)

        # Key sequence to start new game
        pyautogui.keyDown(RESET0)
        pyautogui.press(RESET1)
        pyautogui.keyUp(RESET0)
        pyautogui.keyDown(START)
        sleep(0.1)
        pyautogui.keyUp(START)
        pyautogui.keyDown(START)
        sleep(0.1)
        pyautogui.keyUp(START)
        sleep(11)
        obs, _, _ = Computer_Vision.observe(self.window)
        pyautogui.keyDown(FIRE)
        return obs

    def render(self, mode='human', close=False):
        return None
