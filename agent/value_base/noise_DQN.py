from ..Base import agent_base
import torch
import random
import numpy as np
import torch.nn.functional as F

import torch.nn as nn
import modules

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = modules.noise_layer.NoiseLayer(64, 64)
        self.fc3 = modules.noise_layer.NoiseLayer(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value

class Actor_img(nn.Module):
    def __init__(self, action_dim):
        super(Actor_img, self).__init__()
        self.cnn = modules.cnn.Net()
        self.fc1 = nn.Linear(1568, 800)
        self.fc2 = modules.noise_layer.NoiseLayer(800,128)
        self.fc3 = modules.noise_layer.NoiseLayer(128, action_dim)

    def forward(self, x):
        x= self.cnn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value


class Actor_img2(nn.Module):
    def __init__(self, action_dim):
        super(Actor_img2, self).__init__()
        self.cnn = modules.cnn.Net2()
        self.fc1 = modules.noise_layer.NoiseLayer(3136, 512)
        self.fc2 = modules.noise_layer.NoiseLayer(512, action_dim)

    def forward(self, x):
        x= self.cnn(x)
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x)
        return action_value


class Net(nn.Module):
    def __init__(self, observation_space, action_space,env_observation):
        super(Net, self).__init__()
        self.action_dim = action_space.n
        if env_observation == 'vector' :
            self.obs_dim = observation_space.shape[0]
            self.net = Actor(self.obs_dim, self.action_dim)
        else :
            self.obs_dim = observation_space.shape[-1]
            self.net = Actor_img2(self.action_dim)


    def forward(self, x):
        x = self.net(x)
        return x





class noise_dqn(agent_base) :
    
    def __init__(self,config) :
        agent_base.__init__(self,config)
        self.network = Net(self.observation_space, self.action_space, config.env_observation).to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), \
                lr=self.hyperparameters['lr'])

        self.memory = modules.memory.replay_buffer(config.hyperparameters['batch_size'], config.hyperparameters['buffer_size'])


        self.global_step = 0


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

        Q_ = self.predict(observation)
        ac = torch.argmax(Q_).detach().item()
        return ac


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


        target_Q = self.network(next_states).detach().max(1)[0]
        target_Q = rewards + (self.hyperparameters["discount_rate"] * target_Q * (1 - dones))
        policy_Q = self.network(states)
        
        policy_Q= policy_Q.gather(1, actions.long().unsqueeze(1))
        policy_Q= policy_Q.squeeze(1)
        loss = F.smooth_l1_loss(target_Q, policy_Q)

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
                self.update()

            
        t = self.Timer.finish_episode()

        self.epi_train_time.append(t)
        self.epi_return.append(total_return)
        


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


    def run_n_epi(self,render = True) :
        self.save_model('noise_dqn.{}'.format(0))
        if self.config.max_epi == None :
            self.config.max_epi = 100000
        for epi in range(1,self.config.max_epi) :
            self.step(epi,render = render)

            if epi % self.config.log_interval == 0 :
                print('left train time is',self.left_train_time(epi))
                print('avarage reward is',self.recent_return())
                
            if epi % self.config.save_interval == 0 :
                self.save_model('noise_dqn.{}'.format(epi))


            if epi > 21 :
                return_sum = 0
                for i in range(1,21) : 
                    return_sum += self.epi_return[-i]
                self.epi_avg_return.append(return_sum/20)

    def save_model(self,filename,folder = "./model_save/noise_dqn") :
        torch.save(self.network.state_dict(), f"{folder}/{filename}.pt")

    def load_model(self,filename,folder = "./model_save/noise_dqn") :
        self.network.load_state_dict(torch.load(f"{folder}/{filename}.pt"))