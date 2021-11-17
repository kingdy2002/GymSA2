from agent import policy_base
from agent import value_base
from config.config import Config
from environments.atari import make_atari
from environments.atari import make_atari2
import torch
import gym
import gnwrapper

config_ = Config()

env_name = 'BreakoutDeterministic-v4'
#'Breakout-v0'
#'BreakoutDeterministic-v4'
#CartPole-v1
env = gnwrapper.Animation(make_atari2(env_name,max_episode_steps=10000))
#env =  gnwrapper.Animation(gym.make('CartPole-v1'))
#env = gnwrapper.Animation(gym.make('LunarLander-v2'))
#'Pong-v4'
#env= gym.make('LunarLander-v2')

seed = 7
torch.manual_seed(seed)
env.seed(seed)

config_.env_observation = 'image' #vector, image
config_.env = env
config_.env_name =  env_name
config_.env_args['max_episode_steps'] = 3000
config_.env_args['action_space'] = env.action_space
config_.env_args['observation_space'] = env.observation_space

config_.max_epi = 10001

config_.save_path = 'D:/GymSA/result'

config_.agent_name = 'dqn'
config_.epsilon = True
config_.hyperparameters['batch_size'] = 32
config_.hyperparameters['buffer_size'] = 10000
config_.hyperparameters['lr'] = 1e-4
config_.hyperparameters['discount_rate'] = 0.99


#agent_ = value_base.DQN.dqn(config_) # agent 종류에 맞추어 설정
agent_dqn = value_base.DQN.dqn(config_)
agent_ddqn = value_base.DDQN.ddqn(config_)
agent_per_ddqn = value_base.PER_DDQN.per_ddqn(config_)
agent_noise_dqn = value_base.noise_DQN.noise_dqn(config_)
agent_c51 = value_base.C51.C51(config_)
agent_rainbow = value_base.rainbow.rainbow(config_)