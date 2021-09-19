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
#env =  gnwrapper.Animation(gym.make('CartPole-v1'))
env = gnwrapper.Animation(gym.make('LunarLander-v2'))

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

config_.max_epi = 3000

config_.save_path = 'D:/GymSA/result'

config_.agent_name = 'dqn'
config_.epsilon = True
config_.hyperparameters['batch_size'] = 64
config_.hyperparameters['buffer_size'] = 10000
config_.hyperparameters['lr'] = 1e-4
config_.hyperparameters['discount_rate'] = 0.99


#agent_ = value_base.DQN.dqn(config_) # agent 종류에 맞추어 설정
agent_dqn = value_base.DQN.dqn(config_)
agent_ddqn = value_base.DDQN.ddqn(config_)


agent_per_ddqn = value_base.PER_DDQN.per_ddqn(config_)
agent_noise_dqn = value_base.noise_DQN.noise_dqn(config_)