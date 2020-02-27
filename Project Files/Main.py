import Deep_Q_Learning_Keras as DQL
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import retro

# Constants
ENVIRONMENT             = 'DigDug-Nes'
MOVIE_FILE              = 'example_1.bk2'
WEIGHTS_FILE            = f'dqn_{ENVIRONMENT}_weights.h5f'
CHECKPOINT_WEIGHTS_FILE = f'dqn_{ENVIRONMENT}_weights_checkpoint.h5f'
LOG_FILE                = f'dqn_{ENVIRONMENT}_log.json'

TRAIN       = True
RECORD      = True


def get_training_actions(movie_file=MOVIE_FILE):
    movie = retro.Movie(movie_file)
    movie.step()
    env = retro.make(
        game=movie.get_game(),
        state=None,
        # bk2s can contain any button presses, so allow everything
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
        inttype=retro.data.Integrations.ALL,
    )
    env.initial_state = movie.get_state()
    env.reset()
    actions = []
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        if keys[6] and keys[8]:
            action = 4
        elif keys[7] and keys[8]:
            action = 5
        elif keys[6]:
            action = 1
        elif keys[7]:
            action = 2
        elif keys[8]:
            action = 3
        else:
            action = 0
        actions.append(action)
        # actions.pop(0) to use like a queue
        env.step(keys)
    env.close()
    print(len(actions))
    return actions


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
            nb_steps=2000000,
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




