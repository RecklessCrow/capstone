import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import Deep_Q_Learning_Keras as DQL
import retro
import retrowrapper
from rl.callbacks import FileLogger
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env

# Constants
ENVIRONMENT = 'Galaga-Nes'
PARALLEL_GAMES = 64

if __name__ == '__main__':
    env = retro.make(
        game=ENVIRONMENT,
        use_restricted_actions=retro.Actions.DISCRETE,
        record=True,
        inttype=retro.data.Integrations.ALL
    )
    env.reset()
    DQL.train_DQN_model(env)
    # DQL.train_DQN_model(env, ddqn=True)
    # DQL.train_DQN_model(env, dueling=True)
    # DQL.train_DQN_model(env, ddqn=True, dueling=True)

    # env = make_vec_env(
    #     env_id='Galaga-Nes',
    #     n_envs=PARALLEL_GAMES
    # )
    #
    # model = PPO2(
    #     policy='CnnLstmPolicy',
    #     env=env,
    #     gamma=DQL.GAMMA,
    #     learning_rate=DQL.ALPHA,
    #     nminibatches=PARALLEL_GAMES,
    #     verbose=1,
    #     prioritized_replay=True
    # )
    # model.learn(total_timesteps=DQL.STEPS)
    # model.save(f'./weights/PPO/{ENVIRONMENT}_model')
