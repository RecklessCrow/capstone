import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from rl.callbacks import FileLogger
from PIL import Image, ImageOps
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

# Constants
PIXELS = 128
TARGET_IMAGE_SHAPE = (PIXELS, PIXELS)  # Size we want our image to be for input into CNN in (w, h) for B&W image

ALPHA = 0.05  # Learning rate
GAMMA = 0.99  # Discount rate
EPSILON_MAX = 1.0  # Start value for eps during training
EPSILON_MIN = 0.05  # Lowest value for eps during training
EPSILON_TEST = 0.25

STEPS = 2500000  # Number of actions the NN will take in training
MAX_EXPERIENCES = 1000000  # Max size of replay buffer
EXAMPLE_PERIOD = 10000  # Number actions before NN training kicks in
TARGET_UPDATE = 10000  # Number of actions in an update set
WINDOW_LENGTH = 5  # Number of frames observable in an input
DENSE = 512  # Number of nodes in a dense layer
MEMORY = SequentialMemory(
    limit=MAX_EXPERIENCES,
    window_length=WINDOW_LENGTH
)

# Overwritten class to turn an observation (screenshot) into a resized greyscale image. Also clips the reward to [-1, 1]
class ImageProcessor(Processor):
    def process_observation(self, obs):
        processed_obs = Image.fromarray(obs)

        processed_obs = ImageOps.pad(  # resize and fit to aspect ratio by padding
            image=processed_obs,
            size=TARGET_IMAGE_SHAPE,
            color='black',
            centering=(0.5, 0.5)
        )

        processed_obs = processed_obs.convert('L')  # Greyscale
        processed_obs = np.array(processed_obs)

        assert processed_obs.shape == TARGET_IMAGE_SHAPE  # Make sure image is 1:1 ratio
        return processed_obs.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1, 1)


# Creates a neural network give the number of desired outputs k
def make_model(k):
    model = Sequential()
    model.add(Permute(
        (2, 3, 1),
        input_shape=(WINDOW_LENGTH,) + TARGET_IMAGE_SHAPE
    ))
    model.add(Conv2D(
        32,  # Filters i.e. outputs
        (16, 16),  # Kernal size i.e. size of window inside nn
        strides=(4, 4),
        activation='relu'
    ))
    model.add(Conv2D(
        64,
        (8, 8),
        strides=(2, 2),
        activation='relu'
    ))
    model.add(Conv2D(
        128,
        (4, 4),
        strides=(1, 1),
        activation='relu'
    ))
    model.add(Flatten())
    model.add(Dense(
        units=DENSE,
        activation='relu'
    ))
    model.add(Dense(
        units=k,
        activation='linear'
    ))

    return model


# Makes a Deep
def make_DQN_model(k, ddqn, dueling):
    model = make_model(k)

    processor = ImageProcessor()

    policy = LinearAnnealedPolicy(
        inner_policy=EpsGreedyQPolicy(),
        attr='eps',
        value_max=EPSILON_MAX,
        value_min=EPSILON_MIN,
        value_test=EPSILON_TEST,
        nb_steps=STEPS
    )

    agent = DQNAgent(
        nb_actions=k,
        model=model,
        memory=MEMORY,
        processor=processor,
        policy=policy,
        nb_steps_warmup=EXAMPLE_PERIOD,
        gamma=GAMMA,
        target_model_update=TARGET_UPDATE,
        train_interval=WINDOW_LENGTH,
        enable_double_dqn=ddqn,
        enable_dueling_network=dueling
    )

    agent.compile(
        optimizer=Adam(lr=ALPHA)
    )

    return agent


def train_DQN_model(env, load_file=None, ddqn=False, dueling=False):

    if ddqn and dueling:
        network_type = 'Dueling_DDQN'
    elif ddqn:
        network_type = 'DDQN'
    elif dueling:
        network_type = 'Dueling_DQN'
    else:
        network_type = 'DQN'

    start_time      = datetime.datetime.now().strftime("%m-%d_%H:%M")
    weights_file    = f'./weights/{network_type}/{env.gamename}_weights_{start_time}.h5f'
    log_file        = f'./logs/{network_type}/{env.gamename}_log_{start_time}.json'
    plot_file       = f'./logs/{network_type}/{env.gamename}_training_data_{start_time}.png'

    agent = make_DQN_model(env.action_space.n, ddqn, dueling)

    if load_file is not None:
        agent.load_weights(load_file)

    agent.fit(
        env=env,
        nb_steps=STEPS,
        action_repetition=1,
        callbacks=[FileLogger(log_file)],
        visualize=False,
        verbose=2,
        nb_max_start_steps=10
    )

    agent.save_weights(
        weights_file=weights_file,
        overwrite=True
    )

    log = pd.read_json(log_file)
    plt.grid()
    log.plot(
        kind='scatter',
        x='episode',
        y='episode_reward',
        title='Reward per episode')
    plt.savefig(plot_file)
