from __future__ import print_function

import gym
import gym_wind_turbine


def run_neutral_actions():

    wt = gym.make('WindTurbine-v0')

    # Discovering action and observation spaces
    print(wt.action_space)
    print(wt.observation_space)
    print(wt.action_space.sample())

    print("Episode using neutral action")
    done = False
    wt.reset()
    action = wt.env.neutral_action
    while not done:
        observation, reward, done, info = wt.step(action)

    wt.render()


def run_real_control_test():

    wt = gym.make('WindTurbine-v0')

    print("Episode using real control with constant Wind")
    done = False
    observation = wt.reset()
    accreward = 0
    while not done:
        action = wt.env.real_control(observation)
        observation, reward, done, info = wt.step(action)
        accreward += reward

    print("Accumulated reward: {}".format(accreward))
    wt.render()


def run_random_actions():

    wt = gym.make('WindTurbine-v0')

    print("100 Episodes with random actions")
    for i_episode in range(100):
        print("Episode {}".format(i_episode))
        done = False
        wt.reset()
        while not done:
            action = wt.env.action_space.sample()
            observation, reward, done, info = wt.step(action)
            print("    Action: {}\n    Reward: {}".format(action, reward))
        wt.render()


def run_real_control_actions():

    wt = gym.make('WindTurbineStepwise-v0')

    print("Episode using real control with Stepwise Wind")
    done = False
    observation = wt.reset()
    while not done:
        action = wt.env.real_control(observation)
        observation, reward, done, info = wt.step(action)

    wt.render()


def run_neutral_actions_with_animation():
    wt = gym.make('WindTurbine-v0')
    print("Episode using neutral action with animation")

    wt.env.activate_render_animation()
    wt.render()

    done = False
    wt.reset()
    action = wt.env.neutral_action
    while not done:
        observation, reward, done, info = wt.step(action)

    wt.render(close=True)


if __name__ == '__main__':
    run_neutral_actions()
    run_random_actions()
    run_real_control_actions()
    run_neutral_actions_with_animation()
    # run_real_control_test()

