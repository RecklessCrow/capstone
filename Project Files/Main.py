import Deep_Q_Learning_Keras as DQL
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import retro

# Constants
ENVIRONMENT             = 'Galaga-Nes'
WEIGHTS_FILE            = f'dqn_{ENVIRONMENT}_weights.h5f'
CHECKPOINT_WEIGHTS_FILE = f'dqn_{ENVIRONMENT}_weights_checkpoint.h5f'

TRAIN   = True
LOAD    = False
RECORD  = True

if __name__ == '__main__':

    env = retro.RetroEnv(
        game=ENVIRONMENT,
        use_restricted_actions=retro.Actions.DISCRETE,
        record=RECORD,
        inttype=retro.data.Integrations.ALL
    )

    env.reset()

    agent = DQL.make_DQN_agent(env.action_space.n, TRAIN)

    callbacks = [ModelIntervalCheckpoint(CHECKPOINT_WEIGHTS_FILE, interval=50000)]

    if TRAIN:
        if LOAD:
            agent.load_weights(WEIGHTS_FILE)

        agent.fit(
            env=env,
            nb_steps=2500000,
            action_repetition=1,
            callbacks=callbacks,
            visualize=False,
            log_interval=DQL.TARGET_UPDATE,
        )

        agent.save_weights(WEIGHTS_FILE, overwrite=True)

    else:
        agent.load_weights(WEIGHTS_FILE)
        agent.test(
            env,
            nb_episodes=1,
            visualize=True
        )




