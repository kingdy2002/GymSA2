from agent import policy_base
from agent import value_base
from config.config import Config
from environments.atari import make_atari
import torch
import gym
import gnwrapper

config_ = Config()

env_name = 'Breakout-v0'
#'Breakout-v0'
#CartPole-v1
#env = gnwrapper.Animation(make_atari(env_name,max_episode_steps=10000))
env =  gnwrapper.Animation(gym.make('CartPole-v1'))

#env= gym.make('LunarLander-v2')

seed = 7
torch.manual_seed(seed)
env.seed(seed)

config_.env_observation = 'vector' #vector, image
config_.env = env
config_.env_name =  env_name
config_.env_args['max_episode_steps'] = 10000
config_.env_args['action_space'] = env.action_space
config_.env_args['observation_space'] = env.observation_space

config_.max_epi = 100000
config_.save_path = 'D:/GymSA/result'

config_.agent_name = 'dqn'
config_.epsilon = True
config_.hyperparameters['batch_size'] = 32
config_.hyperparameters['buffer_size'] = 100000
config_.hyperparameters['lr'] = 1.5*10e-4
config_.hyperparameters['discount_rate'] = 0.99


agent_ = value_base.DQN.dqn(config_) # agent 종류에 맞추어 설정
