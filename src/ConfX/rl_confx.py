'''
Two action at a time discrete
'''

import math

import torch.nn as nn

from torch.distributions import Categorical
import numpy as np
import random
import copy


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR_ACTOR = 1e-3 # learning rate of the actor
GAMMA = 0.9  # discount factor
CLIPPING_LSTM = 10
CLIPPING_MODEL = 100
EPISIOLON = 2**(-12)
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Agent():
    def __init__(self, dim_size, resource_size, n_action_steps, action_size, seed, action_step_obs_size=1,lstm_hid_size=128, decay=0.95, is_gemm=False,
                 dropout=0.2,
                 d_model = 10,
                 d_hid = 200,
                 nhead = 2,
                 nlayers = 2,
                ):
        self.dim_size = dim_size
        self.action_size = action_size
        self.n_action_steps = n_action_steps
        self.resource_size = resource_size
        self.action_step_obs_size = action_step_obs_size
        # state_div = [self.dim_size, self.resource_size, self.n_action_steps, self.action_step_obs_size ]
        self.is_gemm = is_gemm
        if is_gemm:
            state_div = [3, 1, 2, 1]
        else:
            state_div = [7,1, 2,1]
        self.state_div =np.cumsum(state_div)
        self.seed = random.seed(seed)

        self.actor = TransformerActor(dim_size, resource_size, n_action_steps, action_size, seed, lstm_hid_size,
                                      dropout=dropout, d_model=d_model, d_hid=d_hid, nhead=nhead, nlayers=nlayers).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.scheduler =optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer, factor=0.9, min_lr=1e-6)
        self.saved_log_probs = []
        self.rewards = []
        self.baseline = None
        self.decay = decay
        self.lstm_hid_size = lstm_hid_size
        self.lstm_hid_value = self.init_hidden_lstm()
        self.lowest_reward = 0
        self.best_reward_whole_eps = float("-Inf")
        self.has_succeed_history = False
        self.bad_counts = 0
    def step(self, state, actions, log_prob, reward, next_state, done, sig, impt, infos):

        self.rewards.append(reward)
        self.saved_log_probs.append(log_prob)
        if done and sig:
            self.learn(GAMMA, impt,infos)

    def save_hidden_lstm(self):
        self.lstm_hid = copy.deepcopy(self.actor.lstm.state_dict())
    def load_hidden_lstm(self):
        self.actor.lstm.load_state_dict(self.lstm_hid)
        del self.lstm_hid
    def init_hidden_lstm(self):
        return None

    def act(self, state, infos, eps=0.0, temperature=1):

        dimensions = state[0:self.state_div[0]]
        action_status = state[self.state_div[0]:self.state_div[1]]
        actions = state[self.state_div[1]:self.state_div[2]]

        action_step = state[self.state_div[2]:self.state_div[3]]

        dimensions = torch.from_numpy(dimensions).type(torch.FloatTensor).to(device)
        action_status = torch.from_numpy(action_status).type(torch.FloatTensor).to(device)
        actions = torch.from_numpy(actions).type(torch.FloatTensor).to(device)
        action_step = torch.from_numpy(action_step).type(torch.LongTensor).to(device)

        (p), self.lstm_hid_value = self.actor(dimensions, action_status,actions, action_step, self.lstm_hid_value,temperature=temperature)
        m = Categorical(p)

        action =m.sample()
        if random.random() < eps:
            action2 = action.data + 1 if random.random() < 0.5 else action -1
            action2 = torch.from_numpy(np.array([action2]))
            action2 = torch.clamp(action2, 0, p.size(1)-1)
            return action2.data, m.log_prob(action2)
        else:
            return action.data, m.log_prob(action)


    def ajust_lr(self, ratio, min_lr=1e-8):
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = max(min_lr, param_group['lr'] * ratio)
    def reset(self):
        self.saved_log_probs = []
        self.rewards = []
        self.lstm_hid_value = self.init_hidden_lstm()
    def set_fitness(self, fitness="energy"):
        self.fitness = fitness

    def minmax_reward(self, reward):
        self.lowest_reward = max(self.lowest_reward, np.min(reward))
        reward -= self.lowest_reward


    def modify_reward(self, reward):
        pick_idx = reward != 0
        pick = reward[pick_idx]
        if len(pick)>0:
            reward[pick_idx] = (reward[pick_idx] - reward[pick_idx].mean())/(reward[pick_idx].std() + EPISIOLON)


    def impt_adj_reward(self, reward, impt):
        if impt is not None:
            reward[:len(impt)] = reward[:len(impt)] * impt
        return reward

    def backup(self, infos):
        if infos["succeed"]:
            self.has_succeed_history = True
            reward_whole_eps = infos["reward_whole_eps"]
            if self.best_reward_whole_eps < reward_whole_eps:
                self.best_reward_whole_eps = reward_whole_eps
                self.actor_backup = copy.deepcopy(self.actor.state_dict())
                self.bad_counts = 0
            else:
                self.bad_counts += 1
        elif self.has_succeed_history:
            self.bad_counts += 1
    def actor_refresh(self,  refresh_threshold=50):
        if self.bad_counts >refresh_threshold:
            self.actor.load_state_dict(self.actor_backup)
            self.bad_counts = 0
    def learn(self, gamma, impt, infos):
        # self.backup(infos)
        rewards = np.array(self.rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + EPISIOLON)
        rewards = self.impt_adj_reward(rewards, impt)
        if self.fitness == "thrpt_btnk":
            self.minmax_reward(rewards)
        dis_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            dis_rewards.insert(0, R)
        dis_rewards = np.array(dis_rewards)
        dis_rewards = (dis_rewards - dis_rewards.mean()) / (dis_rewards.std() + EPISIOLON)


        policy_loss = []
        for log_prob, r in zip(self.saved_log_probs, dis_rewards):
            policy_loss.append(-log_prob * r)
        policy_loss = torch.cat(policy_loss).sum()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), CLIPPING_MODEL)
        # torch.nn.utils.clip_grad_norm_(self.actor.lstm.parameters(), CLIPPING_LSTM)
        self.actor_optimizer.step()
        # self.actor_refresh()
        self.reset()
    def get_chkpt(self):
        chkpt = {"seed":self.seed,
                "actor":self.actor.state_dict(),
                 "baseline": self.baseline,
                 "scheduler": self.scheduler,
                 "optimizer": self.actor_optimizer.state_dict()}
        return chkpt
    def load_actor(self, chkpt):
        self.actor.load_state_dict(chkpt["actor"])
        self.actor_optimizer.load_state_dict(chkpt["optimizer"])
        self.baseline = chkpt["baseline"]
        self.scheduler = chkpt["scheduler"]


def init_weights(m):
    if type(m) == nn.LSTMCell:
        torch.nn.init.orthogonal_(m.weight_hh)
        torch.nn.init.orthogonal_(m.weight_ih)

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class TransformerActor(nn.Module):
    def __init__(self,  dim_size, resource_size, n_action_steps, action_size, seed, 
                 h_size=128, hidden_dim=10, 
                 dropout=0.2,
                 d_model = 10,
                 d_hid = 200,
                 nhead = 2,
                 nlayers = 2,
                ):
        hyperparams = {
            "dropout": dropout,
            "d_model": d_model,
            "d_hid": d_hid,
            "nhead": nhead,
            "nlayers": nlayers,
        }
        print("Transformer hyperparams:", hyperparams)
        seq_len = dim_size + 3
        super(TransformerActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.encoder = nn.Linear(seq_len, seq_len*d_model)
        # self.encoder = nn.Embedding(1000, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(seq_len*1*d_model, 2*action_size) #seq_len*batch_size*d_model x 2*action_size
        self.action_size = action_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.init_weight()

    def init_weight(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, dimension, action_status, action_val, action_step, lstm_hid,temperature=1):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # original code
        # x1 = self.encoder(dimension)
        # x1 = x1.unsqueeze(0)
        # x2 = self.encoder_action(action_val)
        # x2 = x2.unsqueeze(0)
        # x3 = self.encoder_status(action_status)
        # x3 = x3.unsqueeze(0)
        x1 = dimension
        x2 = action_val
        x3 = action_status
        x3 = x3.clamp(min=0)
        src = torch.cat((x1, x2, x3), dim=0)
        # src = src.long(): If encoder is an embedding layer
        src = self.encoder(src).reshape(self.seq_len, 1, self.d_model)
        src = src * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)
        src_mask = generate_square_subsequent_mask(src.size(0)).to(device)
        output = self.transformer_encoder(src, src_mask) # seq_len x batch_size x d_model
        output = output.reshape(1, output.shape[0]*output.shape[1]*output.shape[2])
        output = self.decoder(output)
        output = output.squeeze(0)
        output = output.reshape(2, int(output.shape[0]/2))
        x = F.softmax(output / temperature, dim=1)
        return (x), (None, None)


class Actor(nn.Module):
    def __init__(self,  dim_size, resource_size, n_action_steps, action_size, seed, h_size=128, hidden_dim=10):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.encoder = nn.Linear(dim_size, hidden_dim)
        self.encoder_status = nn.Linear(1, hidden_dim)
        self.encoder_action_range = nn.Linear(action_size, hidden_dim)
        self.encoder_action = torch.nn.Linear(n_action_steps, hidden_dim)
        # self.encoder_action = torch.nn.ModuleList([torch.nn.Embedding(action_size, hidden_dim) for i in range(3)])
        self.fc10 = nn.Linear(resource_size, hidden_dim)
        self.fc11 = nn.Linear(3*hidden_dim, h_size)
        self.fc12 = nn.Linear(h_size, h_size)
        self.fc13 = nn.Linear(h_size, h_size)
        self.fc21 = nn.Linear(h_size, action_size)
        self.fc22 = nn.Linear(h_size, action_size)
        self.fc23 = nn.Linear(h_size, action_size)
        self.output1 = nn.Linear(action_size , action_size)
        self.output2 = nn.Linear(action_size , action_size)
        self.decoder = [self.fc21, self.fc22, self.fc23]
        self.lstm = torch.nn.LSTMCell(h_size, h_size)
        self.n_action_steps = n_action_steps
        # self.init_weight()
    def init_weight(self):
        self.apply(init_weights)
    def forward(self, dimension, action_status, action_val, action_step, lstm_hid,temperature=1):
        print('dimension: ', dimension.shape)
        print('action_status: ', action_status.shape)
        print('action_val: ', action_val.shape)
        print('action_step: ', action_step)
        x1 = self.encoder(dimension)
        x1 = x1.unsqueeze(0)
        x2 = self.encoder_action(action_val)
        x2 = x2.unsqueeze(0)
        x3 = self.encoder_status(action_status)
        x3 = x3.unsqueeze(0)
        x = torch.cat((x1, x2,x3), dim=1)
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        hx, cx = self.lstm(x, lstm_hid)
        x = hx
        decoder_idx = 0
        decoder = self.decoder[decoder_idx]
        x1 = F.relu(decoder(x))
        x1 = self.output1(x1)
        x1 = F.softmax(x1/temperature, dim=1)

        decoder_idx = 1
        decoder = self.decoder[decoder_idx]
        x2 = F.relu(decoder(x))
        x2 = self.output2(x2)
        x2 = F.softmax(x2 / temperature, dim=1)
        x = torch.cat((x1,x2), dim=0)

        return (x), (hx, cx)


