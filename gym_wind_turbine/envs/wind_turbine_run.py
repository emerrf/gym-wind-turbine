import gym
import gym_wind_turbine


def run_neutral_actions():

    wt = gym.make('WindTurbine-v0')

    # Discovering action and observation spaces
    print wt.action_space
    print wt.observation_space
    print wt.action_space.sample()

    print "Episodes using neutral action"
    wt.render()
    observation, _, done, _ = wt.reset()
    action = wt.env.neutral_action
    while not done:
        observation, reward, done, info = wt.step(action)

    wt.render(close=True)


def run_random_actions():

    wt = gym.make('WindTurbine-v0')

    print "100 Episodes with random actions"
    wt.render()
    for i_episode in range(100):
        print "Episode {}".format(i_episode)
        observation, _, done, _ = wt.reset()
        while not done:
            action = wt.env.action_space.sample()
            observation, reward, done, info = wt.step(action)

    wt.render(close=True)


def run_real_control_actions():

    wt = gym.make('WindTurbineStepwise-v0')

    print "Episode using real control with Stepwise Wind"
    wt.render()
    observation, _, done, _ = wt.reset()
    while not done:
        action = wt.env.real_control(observation)
        observation, reward, done, info = wt.step(action)

    wt.render(close=True)


if __name__ == '__main__':
    run_neutral_actions()
    run_random_actions()
    run_real_control_actions()

