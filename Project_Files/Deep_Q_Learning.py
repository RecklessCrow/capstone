import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam

from PIL import Image, ImageOps

from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy

from stable_baselines import DQN

# Constants

# Window information
PIXELS              = 128               # Dimension of window in pixels
TARGET_IMAGE_SHAPE  = (PIXELS, PIXELS)  # Size we want our image to be for input into CNN in (w, h) for B&W image
WINDOW_LENGTH       = 5                 # Number of frames observable in an input

# Model parameters
ALPHA           = 0.001     # Learning rate
GAMMA           = 0.99      # Discount rate
EPSILON_MAX     = 1.0       # Start value for eps during training
EPSILON_MIN     = 0.1       # Lowest value for eps during training
EPSILON_TEST    = 0.05      # Dedicated eps for testing so that the agent (hopefully) cannot get stuck

# Training parameters
STEPS               = 2500000   # Number of actions the NN will take in training
MAX_EXPERIENCES     = 1000000   # Max size of replay buffer
EXAMPLE_PERIOD      = 10000     # Number actions before NN training kicks in
TARGET_UPDATE       = 10000     # Number of actions in an update set
UPDATE_FREQUENCY    = 128       # Number of actions before NN receives an update
DENSE               = 512       # Number of nodes in a dense NN layer
PRE_RAND_ACTIONS    = 10        # Number of random actions agent takes before being recorded

MEMORY = SequentialMemory(
    limit=MAX_EXPERIENCES,
    window_length=WINDOW_LENGTH
)


# Inherited class to turn an observation (screenshot) into a resized greyscale image. Also clips the reward to [-1, 1]
# referenced from https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
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


# Creates a neural network given the number of desired outputs k
def _make_model(k):
    model = Sequential()
    model.add(Permute(
        (2, 3, 1),
        input_shape=(WINDOW_LENGTH,) + TARGET_IMAGE_SHAPE
    ))
    model.add(Conv2D(
        32,  # Filters i.e. outputs
        (16, 16),  # Kernal size i.e. size of window inside nn
        strides=(4, 4),  # Number of times filters are applied
        activation='tanh'
    ))
    model.add(Conv2D(
        64,
        (8, 8),
        strides=(2, 2),
        activation='tanh'
    ))
    model.add(Conv2D(
        128,
        (4, 4),
        strides=(1, 1),
        activation='tanh'
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


# Makes a Deep Q-Learning model using keras-RL
def _make_dqn_model(k, ddqn, dueling):
    model = _make_model(k)

    processor = ImageProcessor()

    policy = LinearAnnealedPolicy(
        inner_policy=EpsGreedyQPolicy(),
        attr='eps',
        value_max=EPSILON_MAX,
        value_min=EPSILON_MIN,
        value_test=EPSILON_TEST,
        nb_steps=STEPS
    )

    test_policy = GreedyQPolicy()

    agent = DQNAgent(
        nb_actions=k,
        model=model,
        memory=MEMORY,
        processor=processor,
        policy=policy,
        test_policy=test_policy,
        nb_steps_warmup=EXAMPLE_PERIOD,
        gamma=GAMMA,
        target_model_update=TARGET_UPDATE,
        train_interval=UPDATE_FREQUENCY,  # number of actions before an update
        enable_double_dqn=ddqn,
        enable_dueling_network=dueling
    )

    agent.compile(
        optimizer=Adam(lr=ALPHA)
    )

    return agent


# Trains a keras DQN model
def train_dqn_model(env, save_file=None, ddqn=False, dueling=False):
    if ddqn and dueling:
        network_type = 'Dueling_DDQN'
    elif ddqn:
        network_type = 'DDQN'
    elif dueling:
        network_type = 'Dueling_DQN'
    else:
        network_type = 'DQN'

    start_time = datetime.datetime.now().strftime("%m-%d_%H:%M")
    weights_file = f'./weights/{network_type}/{env.gamename}_weights_{start_time}.h5f'
    log_file = f'./logs/{network_type}/{env.gamename}_log_{start_time}.json'
    plot_file = f'./logs/{network_type}/{env.gamename}_training_data_{start_time}.png'

    agent = _make_dqn_model(env.action_space.n, ddqn, dueling)

    # File needs to have the same size NN to work
    if save_file is not None:
        agent.load_weights(save_file)

    # Train the agent
    agent.fit(
        env=env,
        nb_steps=STEPS,
        action_repetition=1,
        callbacks=[FileLogger(log_file)],
        verbose=2,
        nb_max_start_steps=PRE_RAND_ACTIONS
    )

    # Save weights, uncertain if loading them does anything
    agent.save_weights(
        weights_file=weights_file,
        overwrite=True
    )

    # Plot the reward over time
    log = pd.read_json(log_file)
    plt.grid()
    log.plot(
        kind='scatter',
        x='episode',
        y='episode_reward',
        title='Reward per episode')
    plt.savefig(plot_file)


# Tests a keras DQN model for ten games
def test_dqn_model(env, save_file, ddqn=False, dueling=False, games=10):
    if save_file is None:
        print('save_file cannot be None')
        return

    agent = _make_dqn_model(env.action_space.n, ddqn, dueling)
    agent.load_weights(save_file)
    agent.test(
        env=env,
        nb_episodes=games,
        nb_max_start_steps=PRE_RAND_ACTIONS
    )


# Trains a DQN model using openai's baseline import
def train_dqn_model_2(env, save_file=None, ddqn=False, dueling=False):
    if save_file is None:
        model = DQN(
            policy='CnnPolicy',
            env=env,
            gamma=GAMMA,
            learning_rate=ALPHA,
            verbose=1,
            exploration_initial_eps=EPSILON_MAX,
            exploration_final_eps=EPSILON_MIN,
            # 1,000,000 observations overflowed my memory (64GB), reduce if its too many
            buffer_size=MAX_EXPERIENCES // 100,
            learning_starts=EXAMPLE_PERIOD,
            target_network_update_freq=TARGET_UPDATE,
            batch_size=WINDOW_LENGTH,
            train_freq=UPDATE_FREQUENCY,
            double_q=ddqn,
            # dueling is a param in the policy, thus must be passed as a dic of parameters
            policy_kwargs=dict(dueling=dueling)
        )
    else:
        # This will load all the params from the previous session (alpha, policy, ect.)
        model = DQN.load(
            load_path=save_file,
            env=env,
        )

    model.learn(
        total_timesteps=STEPS,
        log_interval=10,
    )

    model.save(f'./weights/DQN/{env.gamename}_model')


# Tests a open-ai DQN model for ten games
def test_dqn_model_2(env, save_file):
    model = DQN.load(
        load_path=save_file,
        env=env,
    )

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='human')
