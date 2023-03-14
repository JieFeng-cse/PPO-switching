import numpy as np
import torch
import torch.nn as nn
import gym
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from environment import Hybrid_system


import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = gym.make("BipedalWalker-v3")
        env = Hybrid_system()
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init
# def plot_training_results(training_steps_per_second, reward_averages, reward_std):
#     """
#     Utility function for plotting the results of training

#     :param training_steps_per_second: List[double]       
#     :param reward_averages: List[double]
#     :param reward_std: List[double]
#     """
#     plt.figure(figsize=(9, 4))
#     plt.subplots_adjust(wspace=0.5)
#     plt.subplot(1, 2, 1)
#     plt.errorbar(PROCESSES_TO_TEST, reward_averages, yerr=reward_std, capsize=2, c='k', marker='o')
#     plt.xlabel('Processes')
#     plt.ylabel('Average return')
#     plt.subplot(1, 2, 2)
#     plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
#     plt.xticks(range(len(PROCESSES_TO_TEST)), PROCESSES_TO_TEST)
#     plt.xlabel('Processes')
#     plt.ylabel('Training steps per second')
#     plt.show()

env_id = 'Hybrid'
# The different number of processes that will be used
n_procs = 16
NUM_EXPERIMENTS = 2 # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
TRAIN_STEPS = 50000000
# Number of episodes for evaluation
EVAL_EPS = 20
ALGO = PPO

# We will create one environment to evaluate the agent on
# eval_env = gym.make("BipedalWalker-v3")
eval_env = Hybrid_system()
eval_env.seed(51)
reward_averages = []
reward_std = []
training_times = []
total_procs = 0

print('Running for n_procs = {}'.format(n_procs))
if n_procs == 1:
    # if there is only one process, there is no need to use multiprocessing
    train_env = DummyVecEnv([lambda: gym.make(env_id)])
else:
    # Here we use the "fork" method for launching the processes, more information is available in the doc
    # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
    train_env = SubprocVecEnv([make_env(env_id, i+n_procs+1) for i in range(n_procs)], start_method='fork')

rewards = []
times = []

for experiment in range(NUM_EXPERIMENTS):
    # it is recommended to run several experiments due to variability in results
    train_env.reset()
    model = ALGO('MlpPolicy', train_env, verbose=1,tensorboard_log='./PPO_tensorboard/')
    start = time.time()
    model.learn(total_timesteps=TRAIN_STEPS, tb_log_name="first_run")
    times.append(time.time() - start)
    mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
    rewards.append(mean_reward)
    print(mean_reward)
# Important: when using subprocesses, don't forget to close them
# otherwise, you may have memory issues when running a lot of experiments
train_env.close()
reward_averages.append(np.mean(rewards))
reward_std.append(np.std(rewards))
training_times.append(np.mean(times))

training_steps_per_second = [TRAIN_STEPS / t for t in training_times]

# plot_training_results(training_steps_per_second, reward_averages, reward_std)