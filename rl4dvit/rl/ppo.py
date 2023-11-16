import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10

# env = gym.make('CartPole-v0').unwrapped
# num_state = env.observation_space.shape[0]
# num_action = env.action_space.n
# torch.manual_seed(seed)
# env.seed(seed)
num_state = 768 
num_action = 2
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 4*num_state)
        self.fc2 = nn.Linear(4*num_state, 4*num_state)
        self.action_head = nn.Linear(4*num_state, num_action)

    def forward(self, x):
        # size of x: [token_num, token_dim] -> [197, 768]
        # size of action: [token_num] -> [197]
        x = F.relu(self.fc2(self.fc1(x)))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 4*num_state)
        self.fc2 = nn.Linear(4*num_state, 4*num_state)
        self.state_value = nn.Linear(4*num_state, 1)

    def forward(self, x):
        x = F.relu(self.fc2(self.fc1(x)))
        value = self.state_value(x)
        return value


# class PPO(nn.Module):
class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 1
    buffer_capacity = 12
    batch_size = 4

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('../exp')
        self.reward_one_epoch = 0
        self.reward_one_batch = 0
        self.warm_up_step = 0
        self.total_steps = 0
        self.lr_a = 1e-6
        self.lr_c = 3e-6

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_a)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(),
                                               lr=self.lr_c)
        # self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        # self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')

    def select_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0)
        # size of state:[token_num, token_dim]
        # -> [197, 768]
        with torch.no_grad():
            # size of action_prob:[token_num, action_dim]
            # -> [197, 2]
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        # size of action:[token_num]
        # -> [197]
        action = c.sample()
        # return action, action_prob
        return action.item(), action_prob[:,action.item()].item()

    def select_action_batch(self, state):
        import time 
        d = time.perf_counter()
        with torch.no_grad():
            action_prob = self.actor_net(state)

        b = time.perf_counter()
        batch_size = state.shape[0]
        token_num = state.shape[1]
        # action_batch = np.empty(shape=[0, batch_size])
        # action_prob_batch = np.empty(shape=[0, batch_size])
        # action = np.empty(shape=[0, token_num])

        c_batch = Categorical(action_prob)
        action_batch = c_batch.sample()

        index = action_batch.reshape(action_batch.shape[0], action_batch.shape[1], 1)

        action_prob_batch = torch.gather(action_prob, 2, index)
        action_prob_batch = action_prob_batch.reshape(action_prob_batch.shape[0],
                                  action_prob_batch.shape[1])
        # for i in range(batch_size):
        #     c_image = Categorical(action_prob[i])
        #     action_image = c_image.sample()
        #     action_image = action_image.view(-1,1)

        #     # action_batch = np.append(action_batch, [action_image.cpu().numpy()], axis=0)
        # 
        #     index = action_image.reshape(action_image.shape[0], 1)
        #     action_prob_image = torch.gather(action_prob[i], 1, index)
        #     # action_prob_batch = np.append(action_prob_batch, [action_prob_image.cpu()
        #     #                                                   .numpy()], axis=0)
        #     if i == 0:
        #         action_batch = action_image
        #         action_prob_batch = action_prob_image
        #     else:
        #         action_batch = torch.cat((action_batch, action_image),dim=-1)
        #         action_prob_batch = torch.cat((action_prob_batch, action_prob_image),
        #                                       dim=-1)

        c = time.perf_counter()
            
        # print("run time_1:", (b-d) * 1000)
        # print("run time_2:", (c-b) * 1000)

        # action_prob_batch = torch.index_select(action_prob, 2, action_batch)
        # index = action_batch.reshape(action_batch.shape[0], action_batch.shape[1],1)
        # action_prob_batch = torch.gather(action_prob, 2, index)

        return action_batch, action_prob_batch

    def select_action_image(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0)
        # size of state:[token_num, token_dim]
        # -> [197, 768]
        with torch.no_grad():
            # size of action_prob:[token_n]um, action_dim]
            # -> [197, 2]
            action_prob = self.actor_net(state)

        self.warm_up_step = 0
        if self.warm_up_step < 7680:
            action_prob_a = action_prob[:, 0] * 0.2
            action_prob_b = action_prob[:, 1] + action_prob[:, 0] * 0.8
            action_prob = torch.cat((action_prob_a.view(-1,1),
                                     action_prob_b.view(-1,1)),1)

            self.warm_up_step += 1

        # if self.warm_up_step >= 7680 and self.warm_up_step < 15360:
        #     # prob_min = torch.take(action_prob, action_prob.argmin())
        #     # action_prob_a = action_prob[:,0] - prob_min
        #     # action_prob_b = action_prob[:,1] + prob_min
        #     # action_prob = torch.cat((action_prob_a.view(-1,1),
        #     #                          action_prob_b.view(-1,1)),1)
        #     action_prob_a = action_prob[:, 0] * (0.15 +
        #                                          0.85*(self.warm_up_step/15360))
        #     action_prob_b = action_prob[:, 1] + action_prob[:, 0]*(0.85-0.85*self.warm_up_step/15360)
        #     action_prob = torch.cat((action_prob_a.view(-1,1),
        #                              action_prob_b.view(-1,1)),1)

        #     self.warm_up_step += 1


        c = Categorical(action_prob)
        # size of action:[token_num]
        # -> [197]
        action_batch = c.sample()
        # action_prob_batch = torch.empty(action_prob.shape[0],
        #                                 device=action_batch.device)
        # for i in range(action_batch.shape[0]):
        #     action_prob_batch[i] = action_prob[i, action_batch[i]]

        index = action_batch.reshape(action_batch.shape[0], 1)
        action_prob_batch = torch.gather(action_prob, 1, index)
        return action_batch, action_prob_batch
        # return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        # torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        # torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.actor_net.state_dict(), './param/net_param/actor_net.pkl')
        torch.save(self.critic_net.state_dict(), './param/net_param/critic_net.pkl')
        
    def save_param_best(self):
        torch.save(self.actor_net.state_dict(), './param/net_param/actor_net_best.pkl')
        torch.save(self.critic_net.state_dict(), './param/net_param/critic_net_best.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1


    def update(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).cuda()
        # state = [t.state for t in self.buffer]
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).cuda()
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer],
                                           dtype=torch.float).view(-1, 1).cuda()

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).cuda()
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            # for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))),
                                      len(self.buffer), False):
                # if self.training_step % 1000 ==0:
                    # print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience
        self.lr_decay(self.total_steps)

    def lr_decay(self, total_steps):
        if total_steps <= 1e7:
            lr_a_now = self.lr_a * (1 - total_steps / 1e7)
            lr_c_now = self.lr_c * (1 - total_steps / 1e7)
        else:
            lr_a_now = 0
            lr_c_now = 0
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_net_optimizer.param_groups:
            p['lr'] = lr_c_now

