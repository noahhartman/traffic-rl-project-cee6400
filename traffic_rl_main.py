# traffic_rl_main.py

"""
Main training script for adaptive traffic signal control using PPO
in a SUMO simulation environment with reinforcement learning.
"""

import os
import gym
import numpy as np
import traci
import sumolib
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym import spaces

# ---- Configuration ----
PHASES = ["NS_GREEN", "EW_GREEN", "NS_LEFT", "EW_LEFT"]
STATE_SIZE = 17
ACTION_SIZE = len(PHASES)
EPISODE_STEPS = 1000

# ---- Custom SUMO-Gym Environment ----
class SumoTrafficEnv(gym.Env):
    def __init__(self, reward_type="throughput"):
        super(SumoTrafficEnv, self).__init__()
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Box(low=0, high=100, shape=(STATE_SIZE,), dtype=np.float32)
        self.reward_type = reward_type
        self.current_phase = 0
        self.step_count = 0
        self._start_sumo()

    def _start_sumo(self):
        sumo_binary = sumolib.checkBinary("sumo")
        traci.start([sumo_binary, "-c", "network/single_intersection.sumocfg"])

    def reset(self):
        traci.close()
        self._start_sumo()
        self.step_count = 0
        self.current_phase = 0
        return self._get_state()

    def step(self, action):
        self.current_phase = action
        self._apply_action(action)
        traci.simulationStep()
        self.step_count += 1
        state = self._get_state()
        reward = self._calculate_reward(state)
        done = self.step_count >= EPISODE_STEPS
        return state, reward, done, {}

    def _get_state(self):
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in self._lane_ids()]
        wait_times = [traci.lane.getWaitingTime(lane_id) for lane_id in self._lane_ids()]
        return np.array(queue_lengths + wait_times + [self.current_phase], dtype=np.float32)

    def _apply_action(self, action):
        phase_program = {
            0: "GGrrrrGGrrrr",  # NS_GREEN
            1: "rrGGrrrrGGrr",  # EW_GREEN
            2: "GrrrrGrrrrrr",  # NS_LEFT
            3: "rrGrrrrGrrrr"   # EW_LEFT
        }
        traci.trafficlight.setRedYellowGreenState("TL1", phase_program[action])

    def _lane_ids(self):
        return ["lane_N0", "lane_S0", "lane_E0", "lane_W0", "lane_NL", "lane_SL", "lane_EL", "lane_WL"]

    def _calculate_reward(self, state):
        vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self._lane_ids()])
        wait_time = np.mean(state[8:16])
        if self.reward_type == "throughput":
            return vehicle_count
        elif self.reward_type == "wait":
            return -wait_time
        elif self.reward_type == "hybrid":
            return 0.6 * vehicle_count - 0.4 * wait_time
        else:
            return 0.0

# ---- Training ----
def train_model(reward_type="throughput"):
    env = SumoTrafficEnv(reward_type=reward_type)
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(f"ppo_sumo_{reward_type}")
    env.close()

if __name__ == "__main__":
    for reward_type in ["throughput", "wait", "hybrid"]:
        train_model(reward_type)
