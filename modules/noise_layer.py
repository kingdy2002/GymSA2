import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class NoiseLayer(nn.Module) :
    def __init__(self,in_ch,out_ch,is_train = True):
        super(NoiseLayer,self).__init__()
        self.is_train = is_train
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mu_weight = nn.Parameter(torch.FloatTensor(out_ch, in_ch).to(self.device))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_ch).to(self.device)) 
        self.sigma_weight = nn.Parameter(torch.FloatTensor(out_ch, in_ch).to(self.device))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_ch).to(self.device))

        self.register_buffer("epsilon_weight", torch.FloatTensor(out_ch, in_ch).to(self.device))
        self.register_buffer("epsilon_bias", torch.FloatTensor(out_ch).to(self.device))
        self.reset_par()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()
        
        if self.is_train :
            weight = self.mu_weight + self.sigma_weight.mul(self.epsilon_weight)
            bias = self.mu_bias + self.sigma_bias.mul(self.epsilon_bias)
        else:
            weight = self.mu_weight
            bias = self.mu_bias

        y = F.linear(x, weight, bias)
        
        return y

    def reset_par(self):
        std = 1 / math.sqrt(self.in_ch)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std, std)

        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.in_ch))
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.in_ch))

    def reset_noise(self):
        eps_i = torch.randn(self.in_ch).to(self.device)
        eps_j = torch.randn(self.out_ch).to(self.device)
        eps_b = torch.randn(self.out_ch).to(self.device)
        self.epsilon_i = eps_i.sign() * (eps_i.abs()).sqrt()
        self.epsilon_j = eps_j.sign() * (eps_j.abs()).sqrt()
        self.epsilon_b = eps_b.sign() * (eps_b.abs()).sqrt()
        self.epsilon_weight = self.epsilon_j.ger(self.epsilon_i)
        self.epsilon_bias = self.epsilon_b