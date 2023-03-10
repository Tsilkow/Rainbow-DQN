#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def weight_init(model):
    """Helper function for initializing biases in models"""
    if isinstance(model, nn.Linear):
        nn.init.orthogonal_(model.weight.data)
        model.bias.data.fill_(0.0)


class QNetwork(nn.Module):
    """Network module for estimating Q-values for given actions"""
    def __init__(self, args):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
           nn.Linear(args.state_dim, args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, args.action_dim))
        self.apply(weight_init)
        
    def forward(self, x):
        return self.layers(x)
    

class NoisyLinear(nn.Module):
    """
    Variant of linear module that adds parametrized gaussian noise to 
    linearly calculated results
    """
    def __init__(self, input_size, output_size, std):
        super(NoisyLinear, self).__init__()
        amp = 1/np.sqrt(input_size)
        self.weight_mean = nn.Parameter(
            2*amp * torch.rand((output_size, input_size))
            + torch.full((output_size, input_size), -amp))
        self.weight_var = nn.Parameter(
            torch.full((output_size, input_size), std/np.sqrt(input_size)))
        self.bias_mean = nn.Parameter(
            2*amp * torch.rand((output_size)) 
            + torch.full((output_size, ), -amp))
        self.bias_var = nn.Parameter(
            torch.full([output_size], std/np.sqrt(input_size)))

    def get_device(self):
        return self.weight_mean.device

    def get_noise(self):
        """Method for generating noisy weights and biases"""
        eps_p = torch.randn(self.weight_mean.shape[1], 1)
        eps_b = torch.randn(self.weight_mean.shape[0], 1)
        eps_p = torch.mul(torch.sign(eps_p), torch.sqrt(torch.abs(eps_p)))
        eps_b = torch.mul(torch.sign(eps_b), torch.sqrt(torch.abs(eps_b)))
        weights = self.weight_mean + torch.mul(self.weight_var, (eps_p * eps_b.T).T)
        biases = (self.bias_mean + torch.mul(self.bias_var, eps_b.squeeze())).unsqueeze(0)
        return weights, biases

    def forward(self, x):
        weights, biases = self.get_noise()
        return x @ weights.T + biases
    

class DuelingQNetwork(nn.Module):
    """
    Variant of Q-Network that calculates Q-value by adding together value of 
    given state and advantage given action has over other actions.
    """
    def __init__(self, args, std):
        super(DuelingQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim, std), nn.ReLU(),)
        self.advantage_head = nn.Linear(args.hidden_dim, args.action_dim, std)
        self.value_head = nn.Linear(args.hidden_dim, 1, std)
    
    def forward(self, x):
        headless_output = self.layers(x)
        state_value = self.value_head(headless_output)
        advantages = self.advantage_head(headless_output)
        average_advantage = (advantages.sum(dim=1)/x.shape[1]).unsqueeze(dim=1)
        return state_value + advantages - average_advantage
    

class RainbowQNetwork(nn.Module):
    """
    Variant of Q-Network based on Dueling Q-Network using Noisy Nets. Q-value is
    calculated by adding value of given state to advantage given action has over 
    other actions.
    """
    def __init__(self, args, std):
        super(RainbowQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim), nn.ReLU(),
            NoisyLinear(args.hidden_dim, args.hidden_dim, std), nn.ReLU(),)
        self.advantage_head = NoisyLinear(args.hidden_dim, args.action_dim, std)
        self.value_head = NoisyLinear(args.hidden_dim, 1, std)
    
    def forward(self, x):
        headless_output = self.layers(x)
        state_value = self.value_head(headless_output)
        advantages = self.advantage_head(headless_output)
        average_advantage = (advantages.sum(dim=1)/x.shape[1]).unsqueeze(dim=1)
        return state_value + advantages - average_advantage
