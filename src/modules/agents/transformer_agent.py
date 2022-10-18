import torch
import torch.nn as nn
from .fastformer4 import FastTransformer



from numpy.core.fromnumeric import shape
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.agents.fastformer4 import FastTransformer
#from modules.agents.fastformer3 import FastTransformer
from einops import rearrange


class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.input_shape = input_shape
        self.obs_dim = input_shape - self.n_agents - self.n_actions
        self.act_dim = self.n_agents + self.n_actions
        
        self.max_seq_len = args.batch_size * args.eps_limit
        self.embed_dim = args.embed_dim
        

        
        #self.state_transform = nn.Linear(self.state_dim, args.embed_dim)
        #self.obs_transform = nn.Linear(self.obs_dim, args.embed_dim)
        #self.act_transform = nn.Linear(self.act_dim, args.embed_dim)
        #self.bn1 = nn.BatchNorm1d(self.input_shape)
        #self.transform_obs = nn.Linear(self.obs_dim, args.embed_dim)
        #self.transform_acts = nn.Linear(self.act_dim, args.embed_dim)
        self.transform = nn.Linear(self.input_shape, self.args.embed_dim)
        self.bn = nn.BatchNorm1d(self.args.embed_dim)
        self.bottleneck = nn.Linear(self.embed_dim, self.args.n_actions)

        self.mask = th.ones(1, self.embed_dim).bool().cuda() 
        self.agent = FastTransformer(num_tokens = input_shape, dim = self.embed_dim, #self.embed_dim * 3, 
                                     depth = args.t_depth, max_seq_len = self.max_seq_len, 
                                     absolute_pos_emb=False,
                                     out_dim=self.n_actions, n_agents=args.n_agents,
                                     args=self.args)    
    
    def forward(self, inputs):
        #print('Tmix 2 --> states: {}'.format(states.shape))
        #print("s: {}, agt_out: {}, hist: {}".format(states.shape, agent_qs.shape, histories.shape))
        #
        # s: torch.Size([32, 67, 120]), agt_out: torch.Size([32, 67, 5]), hist: torch.Size([32, 67, 5, 64])
        #

        #states = th.abs(self.state_transform(states.reshape(-1, self.state_dim)))
        #obs = inputs[:, :self.obs_dim]
        #acts = inputs[:, self.obs_dim:]
        #obs = self.transform_obs(obs)
        #acts = self.transform_acts(acts)
        #print('inputs: {}'.format(inputs.shape))
        
        inputs = self.transform(inputs)
        #obs = self.bn(obs)
        #acts = self.bn(acts)
        new_acts = self.agent(inputs) #self.agent(obs, acts)
        #print("new: {}, inp: {}".format(new_acts.shape, inputs.shape))
        new_acts = new_acts.view(-1, self.args.n_actions) + self.bottleneck(inputs)
        return new_acts


        

