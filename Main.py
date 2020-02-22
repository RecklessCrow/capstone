import Deep_Q_Learning_Keras as child
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import retro
import Discretizer

# Constants
ENVIRONMENT = 'Galaga-Nes'
WEIGHTS_FILE            = f'dqn_{ENVIRONMENT}_weights.h5f'
CHECKPOINT_WEIGHTS_FILE = f'dqn_{ENVIRONMENT}_weights_checkpoint.h5f'
LOG_FILE                = f'dqn_{ENVIRONMENT}_log.json'

VISUALIZE = False


if __name__ == '__main__':
    train = True

    env = retro.RetroEnv(game=ENVIRONMENT, inttype=retro.data.Integrations.ALL)
    env = Discretizer.GalagaDiscretizer(env)
    agent = child.make_agent(5)

    if train:

        callbacks = [ModelIntervalCheckpoint(CHECKPOINT_WEIGHTS_FILE, interval=250000)]
        callbacks += [FileLogger(LOG_FILE, interval=100)]

        agent.fit(env, callbacks=callbacks, nb_steps=10000000, log_interval=child.TARGET_UPDATE, visualize=VISUALIZE)

        agent.save_weights(WEIGHTS_FILE, overwrite=True)

    else:
        agent.test(env, nb_episodes=1, visualize=VISUALIZE)
