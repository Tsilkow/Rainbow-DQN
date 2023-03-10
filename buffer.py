#!/usr/bin/env python3

import random
from collections import deque

import numpy as np
import torch
from segment_tree import MinSegmentTree, SumSegmentTree


class ExperienceBuffer:
    """
    Data structure for storing experiences that consist of:
    state -- state of environment
    action -- action taken in this instance
    reward -- reward given as a result of this situation
    next_state -- next state achieved after this one
    terminal -- flag that's 1 for experience that ended the simulation and 0 
    otherwise
    """
    def __init__(self, args):
        self.states = np.zeros((args.buffer_capacity, args.state_dim), dtype=np.float32)
        self.actions = np.zeros((args.buffer_capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((args.buffer_capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((args.buffer_capacity, args.state_dim), dtype=np.float32)
        self.terminals = np.zeros((args.buffer_capacity, 1), dtype=np.int64)
        self.full = False
        self.idx = 0
        self.args = args 
        
    def add(self, state, action, reward, next_state, terminal):
        """
        Method for adding experiences to the buffer. After reaching buffer 
        capacity, experiences are overwritten.
        """
        self.states[self.idx, :] = state
        self.actions[self.idx, :] = action
        self.rewards[self.idx, :] = reward
        self.next_states[self.idx, :] = next_state
        self.terminals[self.idx, :] = 1 if terminal else 0
        self.idx += 1
        if self.idx == self.args.buffer_capacity:
            self.full = True
            self.idx = 0
            
    def sample(self):
        """Method for sampling one batch of experiences."""
        if self.full: 
            idx = np.random.permutation(self.args.buffer_capacity)[:self.args.batch_size]
        else:
            idx = np.random.permutation(self.idx-1)[:self.args.batch_size]
        states = torch.from_numpy(self.states[idx]).to(self.args.device)
        actions = torch.from_numpy(self.actions[idx]).to(self.args.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.args.device)
        next_states = torch.from_numpy(self.next_states[idx]).to(self.args.device)
        terminals = torch.from_numpy(self.terminals[idx]).long().to(self.args.device)
        return states, actions, rewards, next_states, terminals


class NStepBuffer(ExperienceBuffer):
    """
    Extension of Experience Buffer that evaluates reward of given experience
    by looking n past steps and discounting rewards accordingly
    """
    def __init__(self, args, nstep, **kw):
        super(NStepBuffer, self).__init__(args)
        self.memories = deque(maxlen=nstep)
        self.nstep = nstep 
        
    def add(self, state, action, reward, next_state, terminal):
        terminal_ = 1 if terminal else 0 
        memory = (state, action, reward, next_state, terminal_)
        self.memories.append(memory)
        if len(self.memories) >= self.nstep:
            state, action, reward, next_state, terminal = self.get_nstep()
            self.states[self.idx, :] = state
            self.actions[self.idx, :] = action
            self.rewards[self.idx, :] = reward
            self.next_states[self.idx, :] = next_state
            self.terminals[self.idx, :] = terminal
            self.idx += 1
            if self.idx == self.args.buffer_capacity:
                self.full = True
                self.idx = 0
            
    def get_nstep(self):
        reward = 0
        next_state = None
        terminal = 0
        for k, memory in enumerate(self.memories):
            s, a, r, next_s, t = memory
            reward += r * self.args.discount**k
            next_state = next_s
            terminal += t
            if terminal >= 1: 
                next_state = s
                break
        state, action, _, _, _ = self.memories.popleft()
        return state, action, reward, next_state, terminal


class PrioritizedBuffer(ExperienceBuffer):
    """
    Variant of Experience buffer that prioritizes transitions that agent
    estimates the worst.
    """
    def __init__(self, args, alpha, beta, **kw):
        super(PrioritizedBuffer, self).__init__(args, **kw)
        tree_capacity = 1
        while tree_capacity < self.args.buffer_capacity:
            tree_capacity *= 2
        self.beta = beta 
        self.alpha = alpha
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.priority_cap = 1
        
    def add(self, state, action, reward, next_state, terminal):
        self.sum_tree[self.idx] = self.priority_cap ** self.alpha
        self.min_tree[self.idx] = self.priority_cap ** self.alpha
        super().add(state, action, reward, next_state, terminal)
                
    def sample(self):
        idx = self.get_idx()

        states = torch.from_numpy(self.states[idx])
        actions = torch.from_numpy(self.actions[idx])
        rewards = torch.from_numpy(self.rewards[idx])
        next_states = torch.from_numpy(self.next_states[idx])
        terminals = torch.from_numpy(self.terminals[idx])
        
        if self.full: max_weight = self.args.buffer_capacity
        else: max_weight = self.idx-1
        max_weight = max(0.00001, self.min_tree.min(0, max_weight)
                         * self.args.buffer_capacity) ** (-self.beta)
        weights = torch.tensor([self.calculate_weight(i) / max_weight for i in idx])
        return states, actions, rewards, next_states, terminals, idx, weights
    
    def update_priorities(self, idx, priorities):
        """Method for updating importance of experiences"""
        for i, priority in zip(idx, priorities):
            priority = min(priority, self.priority_cap)
            self.sum_tree[i] = priority ** self.alpha
            self.min_tree[i] = priority ** self.alpha
                
    def get_idx(self):
        """Method for getting indices for prioirtized sampling."""
        idxs = []
        range_len = self.sum_tree.sum() / self.args.batch_size
        for i in range(self.args.batch_size):
            mass = (random.random() + i) * range_len
            idx = self.sum_tree.find_prefixsum_idx(mass)
            idxs.append(idx)
        return idxs
    
    def calculate_weight(self, idx):
        """
        Method for calculating weights of given experiences for learning 
        process.
        """
        return (self.args.buffer_capacity*self.sum_tree[idx])**(-self.beta)


class RainbowBuffer(PrioritizedBuffer, NStepBuffer):
    """
    Experience Buffer that combines approaches of Prioritized and N-step
    Experience Buffers. Experiences are prioritized based on how badly they're
    estimated, while the rewards are calculated by looking at last n steps.
    """
    def __init__(self, args, nstep, alpha, beta):
        super(RainbowBuffer, self).__init__(args, alpha=alpha, beta=beta, nstep=nstep)
        
    def add(self, state, action, reward, next_state, terminal):
        terminal_ = 1 if terminal else 0
        memory = (state, action, reward, next_state, terminal_)
        self.memories.append(memory)
        if len(self.memories) >= self.nstep:
            state, action, reward, next_state, terminal = self.get_nstep()
            self.states[self.idx, :] = state
            self.actions[self.idx, :] = action
            self.rewards[self.idx, :] = reward
            self.next_states[self.idx, :] = next_state
            self.terminals[self.idx, :] = terminal
            self.sum_tree[self.idx] = self.priority_cap ** self.alpha
            self.min_tree[self.idx] = self.priority_cap ** self.alpha
            self.idx += 1
            if self.idx == self.args.buffer_capacity:
                self.full = True
                self.idx = 0