from setuptools import setup

setup(
    name='gym_wind_turbine',
    version='0.0.1',
    description='',
    url='',
    author='Emer Rodriguez Formisano',
    author_email='',
    license='Apache License, Version 2.0',
    install_requires=[
        'gym==0.9.2',
        'CCBlade==1.1.1',
        'matplotlib==2.0.0'
    ],
    entry_points={
        'console_scripts': [
            'gwt-run-neutral=gym_wind_turbine.envs.wind_turbine_run:run_neutral_actions',
            'gwt-run-random=gym_wind_turbine.envs.wind_turbine_run:run_random_actions',
            'gwt-run-real-control=gym_wind_turbine.envs.wind_turbine_run:run_real_control_actions',
            'gwt-run-real-control-test=gym_wind_turbine.envs.wind_turbine_run:run_real_control_test',
            'gwt-run-neutral-animation=gym_wind_turbine.envs.wind_turbine_run:run_neutral_actions_with_animation',
        ]
    }
)