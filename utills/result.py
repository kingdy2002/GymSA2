import torch
import time
from torch.utils.tensorboard import SummaryWriter
import utills.timer

class Writer(object) :
    def __init__(self,config) :
        self.config = config
        self.Timer = utills.timer.timer()
        self.agent_name = self.config.agent_name  # dqn, ppo etc
        self.writer = SummaryWriter(log_dir='./runs/{}'.format(self.agent_name + '-' + self.Timer.time_str()))

    def add(self,name,value,episode) :
        self.writer.add_scalar(name, value, episode)