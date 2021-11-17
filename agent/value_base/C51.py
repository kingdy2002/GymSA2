
from ..Base import agent_base
import torch
import random
import numpy as np
import torch.nn.functional as F
import time
import torch.nn as nn
import modules
from torch.utils.tensorboard import SummaryWriter

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim, atoms):
        super(Actor, self).__init__()
        self.cnn = modules.cnn.Net()
        self.fc1 = nn.Linear(state_dim, 800)
        self.fc2 = nn.Linear(800,128)
        self.fc3 = nn.Linear(128, action_dim * atoms)
        self.atoms = atoms
        self.action_dim = action_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        action_value = F.softmax(x.view(-1, self.atoms),-1).view(-1, self.action_dim, self.atoms)
        return action_value

class Actor_img(nn.Module):
    def __init__(self, action_dim,atoms):
        super(Actor_img, self).__init__()
        self.cnn = modules.cnn.Net()
        self.fc1 = nn.Linear(1568, 800)
        self.fc2 = nn.Linear(800, action_dim *atoms)
        self.atoms = atoms
        self.action_dim = action_dim

    def forward(self, x):
        x= self.cnn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_value = F.softmax(x.view(-1, self.atoms),-1).view(-1, self.action_dim, self.atoms)
        return action_value


class Actor_img2(nn.Module):
    def __init__(self, action_dim , atoms):
        super(Actor_img2, self).__init__()
        self.cnn = modules.cnn.Net2()
        self.fc1 = nn.Linear(288, 288)
        self.fc2 = nn.Linear(288, action_dim*atoms)
        self.atoms = atoms
        self.action_dim = action_dim

    def forward(self, x):
        x= self.cnn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_value = F.softmax(x.view(-1, self.atoms),-1).view(-1, self.action_dim, self.atoms)
        return action_value


class CagetorialNet(nn.Module):
    def __init__(self, observation_space, action_space,env_observation, atoms):
        super(CagetorialNet, self).__init__()
        self.action_dim = action_space.n
        if env_observation == 'vector' :
            self.obs_dim = observation_space.shape[0]
            self.net = Actor(self.obs_dim, self.action_dim,atoms)
        else :
            self.obs_dim = observation_space.shape[-1]
            self.net = Actor_img(self.action_dim,atoms)


    def forward(self, x):
        x = self.net(x)
        return x





class C51(agent_base) :
    
    def __init__(self,config) :
        agent_base.__init__(self,config)
        self.atoms = 51
        
        self.network = CagetorialNet(self.observation_space, self.action_space, config.env_observation,self.atoms).to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), \
                lr=self.hyperparameters['lr'])

        self.epsilon = modules.explore.epsilon(self.config.max_epi,self.config.max_eps,self.config.min_eps)


        self.memory = modules.memory.replay_buffer(config.hyperparameters['batch_size'], config.hyperparameters['buffer_size'])
        self.config = config

        self.global_step = 0
        self.writer = SummaryWriter("./runs/c51")
        
        self.Vmin = -10
        self.Vmax = 10

    def get_state(self, observations,env_observation):
        state = np.array(observations)
        if env_observation == 'image' :
            state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state).to(self.device).float()
        return state.unsqueeze(0)

    def predict(self, state):
        with torch.no_grad() :
            self.network.eval()
            Q = self.network(state)
            self.network.train()
        return Q

    def select_action(self,observation,epi):
        eps = self.epsilon.proportional()
        random_ac = random.randint(0,self.action_space.n -1)
        if random.random() < eps :
            ac = random_ac
        else :
            Q_ = self.predict(observation)
            dist = Q_ * torch.linspace(self.Vmin, self.Vmax, self.atoms).to(self.device)

            dist = dist.sum(2)
            
            ac = torch.argmax(dist).detach().item()

        return ac

    def target_distribution(self, target_p , rewards , dones) :
        batch_size = self.config.hyperparameters['batch_size']
        
        d_z = ((self.Vmax - self.Vmin)/(self.atoms - 1))
        support = torch.linspace(self.Vmin,self.Vmax,self.atoms).to(self.device)
        target_Q = target_p * support
        action = target_Q.sum(2).max(1)[1]

        action = action.unsqueeze(1).unsqueeze(1).expand(-1,1,self.atoms)

        target_dis = target_p.gather(1,action).squeeze(1)
        rewards = rewards.unsqueeze(1).expand_as(target_dis)
        dones = dones.unsqueeze(1).expand_as(target_dis)
        support = support.unsqueeze(0).expand_as(target_dis)
        

        Tz = rewards + (1 - dones) * self.hyperparameters["discount_rate"] * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b  = (Tz - self.Vmin) / d_z
        l  = b.floor().long()
        u  = b.ceil().long()
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.atoms).to(self.device)
        proj_dis = torch.zeros(target_dis.size()).to(self.device)
        
        proj_dis.view(-1).index_add_(0, (l + offset).view(-1), (target_dis * (u.float() - b)).view(-1))
        proj_dis.view(-1).index_add_(0, (u + offset).view(-1), (target_dis * (b - l.float())).view(-1))
        

        return proj_dis

    def compute_loss(self,states, actions, rewards,next_states, dones):
        """
        if isinstance(states, list):
            states = torch.tensor([states]).to(self.device)
        if isinstance(actions, list):
            actions = [actions]
        if isinstance(rewards, list):
            rewards = [rewards]
        if isinstance(next_states, list):
            next_states = torch.tensor([next_states]).to(self.device)
        if isinstance(dones, list):
            dones = [dones]
        """
        target_p = self.network(next_states).detach()
        proj_dis = self.target_distribution(target_p,rewards,dones)

        policy_Q = self.network(states)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.atoms)
        policy_Q= policy_Q.gather(1, actions.long()).squeeze(1)

        loss = -(proj_dis * policy_Q.log()).sum(1).mean()
        return loss


    def step(self,epi,render = True) :

        self.Timer.start_episode()
        total_return = 0
        step = 0
        state = self.env.reset()
        env_observation = self.config.env_observation
        state = self.get_state(state,env_observation)
        done = False

        while not done :
            step += 1
            self.global_step += 1

            if step > self.config.env_args['max_episode_steps'] :
                break

            action = self.select_action(state,epi)
            next_state, reward, done, info = self.env.step(action)

            next_state = self.get_state(next_state,env_observation)
            if render : 
                self.env.render()


            reward_ = torch.tensor([reward], device=self.device).float()
            done_ = torch.tensor([done], device=self.device).float()
            action_ = torch.tensor([action], device=self.device).float()



            self.memory.push(state, action_, reward_, next_state,done_)


            
            state = next_state
            total_return = total_return + reward
            if self.global_step % self.config.update_interval == 0 and self.global_step > self.config.train_start :
                loss = self.update()
                self.writer.add_scalar("Loss/train", loss, self.global_step)
        
        t = self.Timer.finish_episode()

        self.epi_train_time.append(t)
        self.epi_return.append(total_return)
        
    def step_test(self,epi,render = True) :

        self.Timer.start_episode()
        total_return = 0
        step = 0
        state = self.env.reset()
        env_observation = self.config.env_observation
        state = self.get_state(state,env_observation)
        done = False

        while not done :
            step += 1
            self.global_step += 1

            if step > self.config.env_args['max_episode_steps'] :
                break
            
            Q_ = self.predict(state)
            action = torch.argmax(Q_).detach().item()

            next_state, reward, done, info = self.env.step(action)
            state = self.get_state(next_state,env_observation)
            if render : 
                self.env.render()



    def update(self) :

        batch = self.memory.make_batch()
        if batch == None :
            return
        states_batch = torch.cat([b.state for b in batch if b is not None]).float().to(self.device)
        actions_batch = torch.cat([b.action for b in batch if b is not None]).float().to(self.device)
        rewards_batch = torch.cat([b.reward for b in batch if b is not None]).float().to(self.device)
        next_states_batch = torch.cat([b.next_state for b in batch if b is not None]).float().to(self.device)
        dones_batch = torch.cat([b.done for b in batch if b is not None]).float().to(self.device)

        loss = self.compute_loss(states_batch,actions_batch,rewards_batch,next_states_batch,dones_batch)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()


    def run_n_epi_test(self,render = True) :
        for epi in range(1,self.config.max_epi) :
            self.step_test(self.config.max_epi,render = render)

    def run_n_epi(self,render = True) :
        self.save_model('c51.{}'.format(0))
        if self.config.max_epi == None :
            self.config.max_epi = 100000
        for epi in range(1,self.config.max_epi) :
            self.step(epi,render = render)

            if epi % self.config.log_interval == 0 :
                print('left train time is',self.left_train_time(epi))
                print('avarage reward is',self.recent_return())
                
            if epi % self.config.save_interval == 0 :
                self.save_model('c51.{}'.format(epi))


            if epi > 21 :
                return_sum = 0
                for i in range(1,21) : 
                    return_sum += self.epi_return[-i]
                self.epi_avg_return.append(return_sum/20)
            self.writer.add_scalar("average_return/train", self.epi_return[-1], epi)
        
        self.writer.close()

    def save_model(self,filename,folder = "./model_save/c51") :
        torch.save(self.network.state_dict(), f"{folder}/{filename}.pt")

    def load_model(self,filename,folder = "./model_save/c51") :
        self.network.load_state_dict(torch.load(f"{folder}/{filename}.pt"))