import gym
from gym import spaces
from ccblade.ccblade import CCAirfoil, CCBlade

import numpy as np
from pkg_resources import resource_filename
from os import path, makedirs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

import multiprocessing as mp
import time
import logging
from datetime import datetime


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class WindGenerator:
    def __init__(self, env_settings):
        self.param = env_settings['wind']
        self.t = 0.0
        self.dt = env_settings['timestep']

        assert 'mode' in self.param

        if self.param['mode'] == 'constant':
            self.current_wind = self.param['speed']
            self.read = self._read_constant

        elif self.param['mode'] == 'stepwise':
            self.current_wind = self.param['speed_range'][0]
            self.target_wind = self.param['speed_range'][1]
            self.step_length = self.param['step_length']
            self.read = self._read_stepwise

        elif self.param['mode'] == 'turbulent':
            raise NotImplementedError

        else:
            raise ValueError('Unknown wind generator mode: {mode}'.format(
                **self.param))

    def _read_constant(self, t):
        return self.param['speed']

    def _read_stepwise(self, t):
        if t > 0.0 and np.round(t, 2) % self.step_length == 0.0:
            diff_wind = np.clip(self.target_wind - self.current_wind,
                                -1.0, 1.0)
            self.current_wind += diff_wind
        return self.current_wind

    def reset(self):
        self.t = 0.0


class WindTurbine(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_settings):
        """
        
        """
        logger.debug(env_settings)
        self.dt = env_settings['timestep']

        # Start CCBlade with NREL 5MW
        self.rotor = self._initialise_nrel_5mw()
        self.nrel_5mw_drivetrain_param = {
            'gear_box_ratio': 97.0,       # 97:1
            'rotor_inertia': 38759228.0,  # kg*m^2
            'generator_inertia': 534.116  # kg*m^2
        }

        # Wind
        self.anamometer = WindGenerator(env_settings)

        # Prepare Gym environment
        obs_space_min_max_limits = np.array([
            [3, 25],          # wind [m/s]
            [0, 7000],        # power [kW]
            [0, 1000],        # thrust [kN]
            [0, 15],          # rotor speed [rpm]
            [0.606, 47.403],  # generator torque [kNm]
            [0, 90]           # collective pitch [deg]
        ])
        self.observation_space = spaces.Box(
            low=obs_space_min_max_limits[:, 0],
            high=obs_space_min_max_limits[:, 1])

        act_space_min_max_limits = np.array([
            [-15.0, 15.0],  # gen. torque rate [kNm/s]
            [-8.0, 8.0]     # pitch rate [deg/s]
        ]) * self.dt
        self.action_space = spaces.Box(
            low=act_space_min_max_limits[:, 0],
            high=act_space_min_max_limits[:, 1])

        self.neutral_action = np.array([0.0, 0.0])

        # Simulation initial values
        self.t = 0.0  # Time
        self.t_max = env_settings['duration']
        self.i = 0
        self.i_max = int(env_settings['duration']/env_settings['timestep'])
        self.ep = 0

        self.gen_torq = 0.606
        self.pitch = 0.0
        self.omega = 6.8464
        self.next_omega = self.omega

        # Reward and score
        self.accum_energy = 0.0
        self.prev_reward = None
        self.game_over = False

        # Render
        self.render_animation = False
        plt.ioff()
        self.run_timestamp = self._get_timestamp()
        # render_animation
        self.plotter_data = mp.Queue()
        self.is_plotter_active = False
        # no render_animation
        self.x_t = np.arange(
            0, env_settings['duration'] + env_settings['timestep'],
            env_settings['timestep'])
        self.y_wind = np.full(self.x_t.shape, np.nan)
        self.y_P = np.full(self.x_t.shape, np.nan)
        self.y_T = np.full(self.x_t.shape, np.nan)
        self.y_omega = np.full(self.x_t.shape, np.nan)
        self.y_gen_torq = np.full(self.x_t.shape, np.nan)
        self.y_pitch = np.full(self.x_t.shape, np.nan)
        self.y_reward = np.full(self.x_t.shape, np.nan)

        # Real Control variables
        self.rc_wind = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25])

        self.rc_gen_torq = np.array(
            [0.606, 2.58, 5.611, 9.686, 14.62, 20.174, 25.51, 31.455,
             40.014, 43.094, 43.094, 43.094, 43.094, 43.094, 43.094,
             43.094, 43.094, 43.094, 43.094, 43.094, 43.094, 43.094,
             43.094])

        self.rc_pitch = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3.823, 6.602, 8.668, 10.45, 12.055,
             13.536, 14.92, 16.226, 17.473, 18.699, 19.941, 21.177, 22.347,
             23.469])

    def _get_timestamp(self):
        return datetime.now().strftime('%Y%m%d%H%M%S')

    def activate_render_animation(self):
        plt.ion()
        self.render_animation = True

    def _initialise_nrel_5mw(self):
        """
        Initialise NREL 5MW 
        
        Load NREL 5MW CCAirfoil data
        Based on CCBlade/test/test_ccblade.py
        
        :return: 
        """
        # geometry
        Rhub = 1.5
        Rtip = 63.0

        r = np.array(
            [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
             28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
             56.1667, 58.9000, 61.6333])
        chord = np.array(
            [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
             3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
        theta = np.array(
            [13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
             6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
        B = 3  # number of blades

        # atmosphere
        rho = 1.225
        mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
        basepath = resource_filename('ccblade',
                                     path.join('data', '5MW_AFFiles'))

        # load all airfoils
        airfoil_types = [0] * 8
        airfoil_types[0] = afinit(path.join(basepath, 'Cylinder1.dat'))
        airfoil_types[1] = afinit(path.join(basepath, 'Cylinder2.dat'))
        airfoil_types[2] = afinit(path.join(basepath, 'DU40_A17.dat'))
        airfoil_types[3] = afinit(path.join(basepath, 'DU35_A17.dat'))
        airfoil_types[4] = afinit(path.join(basepath, 'DU30_A17.dat'))
        airfoil_types[5] = afinit(path.join(basepath, 'DU25_A17.dat'))
        airfoil_types[6] = afinit(path.join(basepath, 'DU21_A17.dat'))
        airfoil_types[7] = afinit(path.join(basepath, 'NACA64_A17.dat'))

        # place at appropriate radial stations
        af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

        af = [0] * len(r)
        for i in range(len(r)):
            af[i] = airfoil_types[af_idx[i]]

        tilt = -5.0
        precone = 2.5
        yaw = 0.0

        # create CCBlade object
        rotor = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                        precone, tilt, yaw, shearExp=0.2, hubHt=90.0)

        return rotor

    def _diff_omega(self, t_aero, t_gen, drivetrain_param):
        """
        
        T_aero - N_gear * T_gen = (I_rotor + N_gear^2 * I_gen) * alpha
        alpha = (T_aero - N_gear * T_gen) / (I_rotor + N_gear^2 * I_gen)
        :return: alpha, angular acceleration [rpm/s^2]
        """
        n_gear = drivetrain_param['gear_box_ratio']
        i_rotor = drivetrain_param['rotor_inertia']
        i_gen = drivetrain_param['generator_inertia']

        alpha = (t_aero - n_gear * t_gen) / (i_rotor + n_gear ** 2 * i_gen)
        return alpha * self.dt * 30/np.pi

    def _step(self, action):
        # Take action
        if self.action_space.contains(action):
            # Apply rate change
            self.gen_torq += action[0]
            self.pitch += action[1]

        # Simulate
        self.omega = self.next_omega
        Uinf = self.anamometer.read(self.t)
        P_aero, T, Q = self.rotor.evaluate([Uinf], [self.omega], [self.pitch])
        P_gen = (self.omega
                 * self.gen_torq
                 * self.nrel_5mw_drivetrain_param['gear_box_ratio']
                 * np.pi / 30)
        observation = np.array([Uinf, P_gen, T[0] / 1e3, self.omega,
                                self.gen_torq, self.pitch])

        # End the episode if actions or observations are out boundaries
        self.game_over = not (self.action_space.contains(action)
            and self.observation_space.contains(observation))

        # Compute reward
        if self.prev_reward is None:
            reward, rew_ctrl = 0.0, 0.0
        else:
            P_weight = 1.0
            T_weight = 0.0
            ctrl_weight = 0.1

            (_, P, T, _, _, _) = observation
            (_, prev_P, prev_T, _, _, _) = self.prev_observation
            
            energy = P * (self.dt/3600.0)
            self.accum_energy += energy

            P_chg_rate = (P - prev_P)/prev_P
            T_chg_rate = (T - prev_T)/prev_T
            ctrl_chg = np.square(action).sum()

            rew_power = P_weight * P_chg_rate
            rew_thrust = T_weight * T_chg_rate
            rew_control = ctrl_weight * ctrl_chg
            rew_alive = 0.05
            
            reward = rew_power - rew_thrust - rew_control + rew_alive
            # print("{} = {} - {} - {} + {}".format(
            #     reward, rew_power, rew_thrust, rew_control, rew_alive))
            # print("{} = {}/{}".format(
            #     P_chg_rate/ctrl_chg, P_chg_rate, ctrl_chg))

        done = False
        if self.game_over:
            # Failure before end of simulation
            # print("Game Over", self.accum_energy, reward)
            reward -= self.accum_energy
            done = True
        elif self.i == self.i_max:
            # End of simulation with no failure
            # print("End of simulation", self.accum_energy, reward)
            reward += self.accum_energy

        self.prev_observation = observation
        self.prev_reward = reward

        # Render data points
        if self.render_animation:
            # Send data to plotter
            if self.is_plotter_active:
                plotter_data_point = (done, {
                    't': self.t,
                    'wind': observation[0],
                    'P': observation[1],
                    'T': observation[2],
                    'omega': observation[3],
                    'gen_torq': observation[4],
                    'pitch': observation[5],
                    'reward': reward
                })
                self.plotter_data.put(plotter_data_point)
        else:
            self.x_t[self.i] = self.t
            self.y_wind[self.i] = observation[0]
            self.y_P[self.i] = observation[1]
            self.y_T[self.i] = observation[2]
            self.y_omega[self.i] = observation[3]
            self.y_gen_torq[self.i] = observation[4]
            self.y_pitch[self.i] = observation[5]
            self.y_reward[self.i] = reward

        # Prepare next iteration
        # Update Omega
        self.next_omega += self._diff_omega(Q[0], self.gen_torq*1e3,
                                            self.nrel_5mw_drivetrain_param)
        # Update time
        self.t += self.dt
        self.i += 1

        return observation, reward, done, {}

    def _reset(self):
        self.anamometer.reset()
        # Simulation initial values
        self.t = 0.0  # Time
        self.i = 0
        self.ep += 1
        self.gen_torq = 0.606
        self.pitch = 0.0
        self.omega = 6.8464
        self.next_omega = self.omega
        self.accum_reward = 0.0  # Accumulated reward

        if not self.render_animation:
            self.y_wind = np.full(self.x_t.shape, np.nan)
            self.y_P = np.full(self.x_t.shape, np.nan)
            self.y_T = np.full(self.x_t.shape, np.nan)
            self.y_omega = np.full(self.x_t.shape, np.nan)
            self.y_gen_torq = np.full(self.x_t.shape, np.nan)
            self.y_pitch = np.full(self.x_t.shape, np.nan)
            self.y_reward = np.full(self.x_t.shape, np.nan)

        # return observation only
        self.accum_energy = 0.0
        self.prev_reward = None
        self.game_over = False
        return self._step(self.neutral_action)[0]

    def _render(self, mode='human', close=False):

        def plot_and_save():
            rout_dir = path.join('gwt_output',
                                 'render_{}'.format(self.run_timestamp))
            rout_filename = 'ep_{}_{}.png'.format(
                "%05d" % self.ep, self._get_timestamp())
            rout_path = path.join(rout_dir, rout_filename)

            # Create render output directory
            try:
                makedirs(rout_dir)
            except OSError:
                if not path.isdir(rout_dir):
                    raise

            # Plot
            fig, (ax_wind,
                  ax_P,
                  ax_T,
                  ax_omega,
                  ax_gen_torq,
                  ax_pitch,
                  ax_reward) = plt.subplots(7, figsize=(8, 12), sharex='all',
                                            tight_layout=True)

            fig.suptitle('gym-wind-turbine')

            ax_wind.set_ylabel('Wind [m/s]')
            line_wind = Line2D(self.x_t, self.y_wind, color='black')
            ax_wind.add_line(line_wind)
            ax_wind.set_xlim(0, self.t_max)
            ax_wind.set_ylim(0, 25)
            ax_wind.grid(linestyle='--', linewidth=0.5)

            ax_P.set_ylabel('Power [kW]')
            line_P = Line2D(self.x_t, self.y_P, color='black')
            ax_P.add_line(line_P)
            ax_P.set_xlim(0, self.t_max)
            ax_P.set_ylim(0, 7000)
            ax_P.grid(linestyle='--', linewidth=0.5)

            ax_T.set_ylabel('Thrust [kN]')
            line_T = Line2D(self.x_t, self.y_T, color='black')
            ax_T.add_line(line_T)
            ax_T.set_xlim(0, self.t_max)
            ax_T.set_ylim(0, 1000)
            ax_T.grid(linestyle='--', linewidth=0.5)

            ax_omega.set_ylabel('Rotor speed [rpm]')
            line_omega = Line2D(self.x_t, self.y_omega, color='black')
            ax_omega.add_line(line_omega)
            ax_omega.set_xlim(0, self.t_max)
            ax_omega.set_ylim(0, 15)
            ax_omega.grid(linestyle='--', linewidth=0.5)

            ax_gen_torq.set_ylabel('Gen. Torque [kNm]')
            line_gen_torq = Line2D(self.x_t, self.y_gen_torq, color='blue')
            ax_gen_torq.add_line(line_gen_torq)
            ax_gen_torq.set_xlim(0, self.t_max)
            ax_gen_torq.set_ylim(0.606, 47.403)
            ax_gen_torq.grid(linestyle='--', linewidth=0.5)

            ax_pitch.set_ylabel('Coll. pitch [deg]')
            line_pitch = Line2D(self.x_t, self.y_pitch, color='blue')
            ax_pitch.add_line(line_pitch)
            ax_pitch.set_xlim(0, self.t_max)
            ax_pitch.set_ylim(0, 90)
            ax_pitch.grid(linestyle='--', linewidth=0.5)

            ax_reward.set_ylabel('Reward [units]')
            line_reward = Line2D(self.x_t, self.y_reward, color='green')
            ax_reward.add_line(line_reward)
            ax_reward.set_xlim(0, self.t_max)
            #ax_reward.set_ylim(-200, 5600)
            ax_reward.set_ylim(-1, 1)
            ax_reward.grid(linestyle='--', linewidth=0.5)
            ax_reward.set_xlabel('Time [s]')

            logger.info("Saving figure: {}".format(rout_path))
            plt.savefig(rout_path, dpi=72)
            # plt.close(fig)
            # plt.show()

        if mode == 'human':
            if self.render_animation:
                if not close and not self.is_plotter_active:
                    self.is_plotter_active = True
                    self.p = mp.Process(target=plotter,
                                        args=(self.plotter_data, ))
                    self.p.start()
                else:
                    if self.is_plotter_active:
                        logger.info("Waiting for plotter...")
                        while not self.plotter_data.empty():
                            # logger.info("sleeping")
                            time.sleep(3)
                        # self.p.join()
                        self.p.terminate()
                        self.is_plotter_active = False
            else:
                if close:
                    return None
                else:
                    plot_and_save()

        elif mode == 'rgb_array':
            # Matplotlib to RGB array (gif)
            raise NotImplementedError
        else:
            # Print string
            raise NotImplementedError

    def real_control(self, obs):

        (obs_wind, _, _, _, obs_gen_torq, obs_pitch) = obs

        opt_gen_torq = np.interp(obs_wind, self.rc_wind, self.rc_gen_torq)
        opt_pitch = np.interp(obs_wind, self.rc_wind, self.rc_pitch)

        act_gen_torq_low, act_pitch_low = self.action_space.low
        act_gen_torq_high, act_pitch_high = self.action_space.high
        real_control_action = np.array([
            np.clip(opt_gen_torq - obs_gen_torq, act_gen_torq_low,
                    act_gen_torq_high),
            np.clip(opt_pitch - obs_pitch, act_pitch_low, act_pitch_high)])
        return real_control_action


def plotter(q_plotter_data):

    initial_xmax = 60.0

    fig, (ax_wind,
          ax_P,
          ax_T,
          ax_omega,
          ax_gen_torq,
          ax_pitch) = plt.subplots(6, figsize=(8, 12), sharex='all',
                                   tight_layout=True)

    fig.suptitle('gym-wind-turbine')

    x_t = []
    y_wind = []
    y_P = []
    y_T = []
    y_omega = []
    y_gen_torq = []
    y_pitch = []
    # y_reward = []

    ax_wind.set_ylabel('Wind [m/s]')
    line_wind = Line2D([], [], color='black')
    ax_wind.add_line(line_wind)
    ax_wind.set_xlim(0, initial_xmax)
    ax_wind.set_ylim(0, 25)
    ax_wind.grid(linestyle='--', linewidth=0.5)

    ax_P.set_ylabel('Power [kW]')
    line_P = Line2D([], [], color='black')
    ax_P.add_line(line_P)
    ax_P.set_xlim(0, initial_xmax)
    ax_P.set_ylim(0, 7000)
    ax_P.grid(linestyle='--', linewidth=0.5)

    ax_T.set_ylabel('Thrust [kN]')
    line_T = Line2D([], [], color='black')
    ax_T.add_line(line_T)
    ax_T.set_xlim(0, initial_xmax)
    ax_T.set_ylim(0, 1000)
    ax_T.grid(linestyle='--', linewidth=0.5)

    ax_omega.set_ylabel('Rotor speed [rpm]')
    line_omega = Line2D([], [], color='black')
    ax_omega.add_line(line_omega)
    ax_omega.set_xlim(0, initial_xmax)
    ax_omega.set_ylim(0, 15)
    ax_omega.grid(linestyle='--', linewidth=0.5)

    ax_gen_torq.set_ylabel('Gen. Torque [kNm]')
    line_gen_torq = Line2D([], [], color='blue')
    ax_gen_torq.add_line(line_gen_torq)
    ax_gen_torq.set_xlim(0, initial_xmax)
    ax_gen_torq.set_ylim(0.606, 47.403)
    ax_gen_torq.grid(linestyle='--', linewidth=0.5)

    ax_pitch.set_ylabel('Coll. pitch [deg]')
    line_pitch = Line2D([], [], color='blue')
    ax_pitch.add_line(line_pitch)
    ax_pitch.set_xlim(0, initial_xmax)
    ax_pitch.set_ylim(0, 90)
    ax_pitch.grid(linestyle='--', linewidth=0.5)
    ax_pitch.set_xlabel('Time [s]')

    # ax_reward.set_ylabel('Total Reward [units]')
    # line_reward = Line2D([], [], color='blue')  # 00529F, red: #A2214B
    # ax_reward.add_line(line_reward)
    # ax_reward.set_xlim(0, initial_xmax)
    # ax_reward.set_ylim(-200, 5600)
    # ax_reward.grid(linestyle='--', linewidth=0.5)

    def update_line(i):
        done, data_point = q_plotter_data.get()
        # logger.debug("Plot data point: {}".format(obj))

        x_t.append(data_point['t'])
        y_wind.append(data_point['wind'])
        y_P.append(data_point['P'])
        y_T.append(data_point['T'])
        y_omega.append(data_point['omega'])
        y_gen_torq.append(data_point['gen_torq'])
        y_pitch.append(data_point['pitch'])
        # y_reward.append(data_point['reward'])

        xmin, xmax = ax_wind.get_xlim()
        xlim_mult = 2.0
        if data_point['t'] >= xmax:
            ax_wind.set_xlim(xmin, xmax * xlim_mult)
            ax_wind.figure.canvas.draw()
            ax_P.set_xlim(xmin, xmax * xlim_mult)
            ax_P.figure.canvas.draw()
            ax_T.set_xlim(xmin, xmax * xlim_mult)
            ax_T.figure.canvas.draw()
            ax_omega.set_xlim(xmin, xmax * xlim_mult)
            ax_omega.figure.canvas.draw()
            ax_gen_torq.set_xlim(xmin, xmax * xlim_mult)
            ax_gen_torq.figure.canvas.draw()
            ax_pitch.set_xlim(xmin, xmax * xlim_mult)
            ax_pitch.figure.canvas.draw()
            # ax_reward.set_xlim(xmin, xmax * xlim_mult)
            # ax_reward.figure.canvas.draw()

        if done:
            del x_t[:]
            del y_wind[:]
            del y_P[:]
            del y_T[:]
            del y_omega[:]
            del y_gen_torq[:]
            del y_pitch[:]
            # del y_reward[:]

        else:
            line_wind.set_data(x_t, y_wind)
            line_P.set_data(x_t, y_P)
            line_T.set_data(x_t, y_T)
            line_omega.set_data(x_t, y_omega)
            line_gen_torq.set_data(x_t, y_gen_torq)
            line_pitch.set_data(x_t, y_pitch)
            # line_reward.set_data(x_t, y_reward)

        return [line_wind, line_P, line_T, line_omega, line_gen_torq,
                line_pitch]  # ,line_reward]

    ani = FuncAnimation(fig, update_line, interval=25,
                        blit=True, save_count=25*20*120)
    # ani.save('ani3.gif', writer='imagemagick', fps=20)
    # ani.save(filename='ani3.mp4', fps=20)
    plt.show()
