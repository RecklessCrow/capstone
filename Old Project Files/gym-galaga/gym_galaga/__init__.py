from gym.envs.registration import register

register(
    id='Galaga-v0',
    entry_point='gym_galaga.envs:GalagaEnv',
)