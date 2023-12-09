import numpy as np
import torch

class ReplayBuffer():
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode = args.episode
        self.episode_step = args.episode_step
        self.episode_num = 0 
        self.total_step = 0
        self.buffer = None

    def reset_buffer(self):
        self.buffer = {
            'obs_n': np.empty([self.episode, self.episode_step, self.N, self.obs_dim]),
            'v_n': np.empty([self.batch_size, self.episode_limit + 1, self.N]),
            'obs_n_': np.empty([self.episode, self.episode_step, self.N, self.obs_dim]),
            'cls_token': np.empty([self.episode, self.episode_step, self.obs_dim]),
            'a_n': np.empty([self.episode, self.episode_step, self.N]),
            'a_logprob_n': np.empty([self.episode, self.episode_step, self.N]),
            'r_n': np.empty([self.episode, self.episode_step, self.N]),
            'done_n': np.empty([self.episode, self.episode_step, self.N])
        }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, v_n, obs_n_, cls_token, a_n,
                         a_logprob_n, r_n, done_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['obs_n_'][self.episode_num][episode_step] = obs_n_
        self.buffer['cls_token'][self.episode_num][episode_step] = cls_token
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n
        self.total_step += 1
    
    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key =='a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch
