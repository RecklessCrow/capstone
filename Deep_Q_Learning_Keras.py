from keras.models import Sequential
from keras.layers import ConvLSTM2D, Conv2D, Dense, Flatten, Permute
from keras.optimizers import Adam, nadam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from PIL import Image, ImageOps
import numpy as np

# Constants
TARGET_IMAGE_SHAPE = (128, 128)  # Size we want our image to be for input into CNN in (w, h)

ALPHA = 0.00025  # Learning rate
GAMMA = 0.99  # Discount rate
EPSILON_MAX = 1.0  # Probability of a random action
EPSILON_MIN = 0.1
EPSILON_TEST = 0.5

MAX_EXPERIENCES = 1000000  # Max size of replay buffer
EXAMPLE_PERIOD   = int(MAX_EXPERIENCES / 10)  # Number of actions before observation network gets updated
TARGET_UPDATE = int(EXAMPLE_PERIOD / 2)  # Min size before training
WINDOW_LENGTH   = 5  # Number of frames observable in an input

DENSE = 256


class ImageProcessor(Processor):
    def process_observation(self, obs):
        obs = Image.fromarray(obs)

        processed_obs = ImageOps.pad(  # resize and fit to aspect ratio by padding
            image=obs,
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
        return np.clip(reward, -1., 1.)


def make_model(k):
    model = Sequential()
    model.add(Permute(
        (2, 3, 1),
        input_shape=(WINDOW_LENGTH,) + TARGET_IMAGE_SHAPE
    ))
    # self.model.add(ConvLSTM2D())
    model.add(Conv2D(
        32,  # Filters i.e. outputs
        (8, 8),  # Kernal size i.e. size of window inside nn
        strides=(4, 4),
        activation='relu'
    ))
    model.add(Conv2D(
        64,
        (3, 3),
        strides=(2, 2),
        activation='relu'
    ))
    model.add(Conv2D(
        64,
        (2, 2),
        strides=(1, 1),
        activation='relu'
    ))
    model.add(Flatten())
    model.add(Dense(
        units=DENSE,
        activation='relu'
    ))
    model.add(Dense(
        units=DENSE * 2,
        activation='relu'
    ))
    model.add(Dense(
        units=DENSE,
        activation='relu'
    ))
    model.add(Dense(
        units=k,
        activation='linear'
    ))

    print(model.summary())

    return model


def make_agent(k):
    model = make_model(k)

    memory = SequentialMemory(
        limit=MAX_EXPERIENCES,
        window_length=WINDOW_LENGTH
    )

    processor = ImageProcessor()

    policy = LinearAnnealedPolicy(
        inner_policy=MaxBoltzmannQPolicy(),
        attr='eps',
        value_max=EPSILON_MAX,
        value_min=EPSILON_MIN,
        value_test=EPSILON_TEST,
        nb_steps=MAX_EXPERIENCES
    )

    agent = DQNAgent(
        model=model,
        nb_actions=k,
        memory=memory,
        processor=processor,
        policy=policy,
        nb_steps_warmup=EXAMPLE_PERIOD,
        gamma=GAMMA,
        target_model_update=TARGET_UPDATE,
        train_interval=4,
        delta_clip=1.0
    )

    agent.compile(
        optimizer=Adam(lr=ALPHA),
        metrics=['mae']
    )

    return agent
