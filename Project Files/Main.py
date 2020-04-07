import Deep_Q_Learning_Keras as DQL
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import retro

# Constants
ENVIRONMENT             = 'Galaga-Nes'
WEIGHTS_FILE            = f'dqn_{ENVIRONMENT}_weights.h5f'
CHECKPOINT_WEIGHTS_FILE = f'dqn_{ENVIRONMENT}_weights_checkpoint.h5f'
LOG_FILE                = f'dqn_{ENVIRONMENT}_log.json'

TRAIN       = True
RECORD      = True

if __name__ == '__main__':

    env = retro.RetroEnv(
        game=ENVIRONMENT,
        use_restricted_actions=retro.Actions.DISCRETE,
        record=RECORD,
        inttype=retro.data.Integrations.ALL
    )

    env.reset()

    agent = DQL.make_DQN_agent(env.action_space.n, TRAIN)

    if TRAIN:

        callbacks = [ModelIntervalCheckpoint(CHECKPOINT_WEIGHTS_FILE, interval=250000)]
        callbacks += [FileLogger(LOG_FILE, interval=100)]

        agent.fit(
            env=env,
            nb_steps=5000000,
            action_repetition=1,
            callbacks=callbacks,
            visualize=False,
            log_interval=DQL.TARGET_UPDATE,
        )

        agent.test(
            env,
            nb_episodes=10,
            visualize=True
        )

        agent.save_weights(WEIGHTS_FILE, overwrite=True)

    else:
        agent.load_weights(WEIGHTS_FILE)
        agent.test(
            env,
            nb_episodes=1,
            visualize=True
        )




