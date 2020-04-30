# This helps get rid of large red text output to the console at runtime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import Deep_Q_Learning as DQL
import multiprocessing
import retro
import retrowrapper

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env

# Constants
ENVIRONMENT = 'Galaga-Nes'

# Set to...
# 0: Train DQN model with keras
# 1: Train DQN model with baseline
# 2: Test keras DQN
# 3: Test baseline DQN
# 4: Train PPO model
# 5: Test PPO model
SELECTION = 2

# Set these for whatever model you're training/testing
DDQN = True  # enable double deep q-networks
DUELING = True  # enable dueling


def train_dqn(env, keras_model):

    env.reset()

    if keras_model:
        # Make Deep Q-Network models using keras-rl
        DQL.train_dqn_model(
            env=env,
            save_file=None,  # Give filepath to weights you would like to load into the model to retrain
            ddqn=DDQN,  # Set to True to enable double deep Q-Network
            dueling=DUELING  # Set to True to enable dueling
        )
    else:
        # Make Deep Q-Network models using Open-ai baseline
        DQL.train_dqn_model_2(
            env=env,
            save_file=None,
            ddqn=DDQN,
            dueling=DUELING
        )


def test_dqn(env, keras):
    if DDQN and DUELING:
        network_type = 'Dueling_DDQN'
    elif DDQN:
        network_type = 'DDQN'
    elif DUELING:
        network_type = 'Dueling_DQN'
    else:
        network_type = 'DQN'
    env.reset()

    if keras:
        # Test the keras model
        DQL.test_dqn_model(
            env=env,
            # Add the appropriate date and time to the end of this file to get the weights of the model trained
            # Or rename the weights file to {ENVIRONMENT}_weights
            save_file=f'/home/cj/PycharmProjects/capstone/Project Files/weights/{network_type}/{ENVIRONMENT}_weights.h5f',
            ddqn=DDQN,
            dueling=DUELING
        )
    else:
        # Test the baseline model
        DQL.test_dqn_model_2(
            env=env,
            save_file=f'./weights/{network_type}/{ENVIRONMENT}_weights'
        )


def train_ppo(env, load=False):

    model = PPO2(
        policy='CnnPolicy',
        env=env,
        gamma=DQL.GAMMA,
        learning_rate=DQL.ALPHA,
        nminibatches=NUM_GAMES // 4,
        verbose=1,
    )

    # Load a model
    # Uses all parameters from previous training
    if load:
        model = PPO2.load(
            load_path=f'./weights/PPO/{ENVIRONMENT}_model',
            env=env,
        )

    model.learn(
        # Training is accelerated on this model thus we need to multiply by 10
        # so that it has time to actually train
        total_timesteps=DQL.STEPS * 10,
    )

    model.save(f'./weights/PPO/{ENVIRONMENT}_model')


def test_ppo():
    env = make_vec_env(
        env_id=retrowrapper.RetroWrapper(
            game=ENVIRONMENT,
            use_restricted_actions=retro.Actions.DISCRETE,
            inttype=retro.data.Integrations.ALL
        ),
        n_envs=1
    )

    model = PPO2.load(
        load_path=f'./weights/PPO/{ENVIRONMENT}_model',
        env=env,
    )

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='human')


if __name__ == '__main__':
    if SELECTION < 4:
        # Create a retro gym env
        env = retro.make(
            game=ENVIRONMENT,
            # Use discrete action space so that we can use a linear output in the NN
            use_restricted_actions=retro.Actions.DISCRETE,
            record=True,
            # Use ALL integration so that retro can detect custom gym environments
            inttype=retro.data.Integrations.ALL
        )
    else:
        if SELECTION == 4:
            NUM_GAMES = multiprocessing.cpu_count()  # Use the number of CPU cores to maximise computer utilization
        else:
            NUM_GAMES = 1
        # create a vector environment filled with
        env = make_vec_env(
            env_id=retrowrapper.RetroWrapper(
                game=ENVIRONMENT,
                use_restricted_actions=retro.Actions.DISCRETE,
                inttype=retro.data.Integrations.ALL
            ),
            n_envs=NUM_GAMES,
            monitor_dir='./logs/PPO/',
        )

    if SELECTION == 0:
        train_dqn(env, True)
    elif SELECTION == 1:
        train_dqn(env, False)
    elif SELECTION == 2:
        test_dqn(env, True)
    elif SELECTION == 3:
        test_dqn(env, False)
    elif SELECTION == 4:
        train_ppo(env, False)
    elif SELECTION == 5:
        test_ppo()

