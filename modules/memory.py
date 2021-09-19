from collections import namedtuple, deque
import torch
import random
from .Maxheap import *

class replay_buffer(object):

    def __init__(self,batch_size,buffer_size) :
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("experience", field_names=["state", "action",\
                                                                "reward", "next_state", "done"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size  = batch_size

    def push(self,states, actions, rewards, next_states, dones) :
        exp = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(exp)

    def make_batch(self) :
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            return batch

        else :
            return None



    def __len__(self) :
        return len(self.memory)

class per (object) :
    def __init__(self,batch_size,buffer_size,alpha = 0.4 , beta = 0.4) :
        self.td_sum = 0
        self.td_buffer = deque(maxlen=buffer_size)
        self.buffer_size =buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("experience", field_names=["state", "action",\
                                                                "reward", "next_state", "done"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size  = batch_size

        self.alpha = alpha
        self.beta = beta


    def push(self,states, actions, rewards, next_states, dones , loss) :
        exp = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(exp)
        loss = loss ** self.alpha
        if len(self.memory) == self.buffer_size :
            self.td_sum -= self.td_buffer[0]

        self.td_buffer.append(loss)
        self.td_sum += loss

    def make_batch(self) :
        td_buffer = np.array(self.td_buffer)
        self.td_sum = td_buffer.sum()
        p = np.array(td_buffer) / self.td_sum
        index = np.random.choice(range(len(self.td_buffer)), size=self.batch_size, replace=False, p=p)
        batch = [self.memory[i] for i in index ]
        td_batch = [self.td_buffer[i] for i in index ]
        w =self.important_weigh(td_batch)
        return batch, w

    def important_weigh(self,td_batch) :
        w = [((1.0 / len(self.td_buffer)) * (self.td_sum / td)) ** self.beta for td in td_batch]
        max_w = max(w)
        w = [i / max_w for i in w]
        w = torch.tensor(w).float().to(self.device)
        return w

    def __len__(self) :
        return len(self.memory)