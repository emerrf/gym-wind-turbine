from gym.envs.registration import register


register(
    id='WindTurbine-v0',
    entry_point='gym_wind_turbine.envs:WindTurbine',
    max_episode_steps=int(60.0/(1.0/20)),  # 60s -> 1200 steps
    kwargs={
        'env_settings': {
            'timestep': 1.0/20.0,
            'duration': 60.0,
            'wind': {
                'mode': 'constant',
                'speed': 8.0,
            }
        }
    }
)

register(
    id='WindTurbineStepwise-v0',
    entry_point='gym_wind_turbine.envs:WindTurbine',
    max_episode_steps=int(120.0/(1.0/20)),  # 120s -> 2400 steps
    kwargs={
        'env_settings': {
            'timestep': 1.0/20.0,
            'duration': 120.0,
            'wind': {
                'mode': 'stepwise',
                'speed_range': [4.0, 24.0],
                'step_length': 5.0
            }
        }
    }
)

# register(
#     id='WindTurbineTurbulent-v0',
#     entry_point='gym_wind_turbine.envs:WindTurbine',
# )
