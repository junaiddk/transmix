
from numpy.core.fromnumeric import shape
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from modules.agents.fastformer2 import FastTransformer # transformer for separate inputs
from modules.agents.fastformer4 import FastTransformer # transformer for combined/concate input 
#from modules.agents.fastformer3 import FastTransformer # transformer for concate states and obs, and separate agent_qs 
from einops import rearrange


class TMixer(nn.Module):
    def __init__(self, scheme, input_shape, args):
        super(TMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.scheme = scheme
        #print(scheme)
        self.input_shape = input_shape
        self.state_dim = int(np.prod(args.state_shape))
        self.n_actions = args.n_actions
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.max_seq_len = args.batch_size * args.eps_limit
        self.embed_dim = args.embed_dim
        self.obs_dim = self.n_agents * self.scheme['obs']['vshape']
        self.gs_dim = self.obs_dim + self.state_dim

        #dim = self.state_dim + self.obs_dim + self.n_agents
        
        #self.state_transform = nn.Linear(self.state_dim, args.embed_dim)
        self.state_transform = nn.Linear(self.gs_dim, args.embed_dim)
        self.aqs_transform = nn.Linear(self.n_agents, args.embed_dim)
        #self.obs_transform = nn.Linear(self.obs_dim, args.embed_dim)

        self.mask = th.ones(1, self.embed_dim).bool().cuda() 
        self.mixer = FastTransformer(num_tokens = input_shape, dim = self.embed_dim * 2, #self.embed_dim * 3, 
                                     depth = args.t_depth, max_seq_len = self.max_seq_len, 
                                     absolute_pos_emb=False,
                                     out_dim=1, n_agents=args.n_agents,
                                     args=self.args)    
    def calc_v(self, agent_qs):
        
        #agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1, keepdim=True)
        #v_tot = rearrange(v_tot, "d -> () d")
        #v_tot = self.
        return v_tot

    def forward(self, agent_qs, obs, states, b_max=0):
        #print('Tmix 2 --> states: {}'.format(states.shape))
        #print("s: {}, agt_out: {}, hist: {}".format(states.shape, agent_qs.shape, histories.shape))
        #
        # s: torch.Size([32, 67, 120]), agt_out: torch.Size([32, 67, 5]), hist: torch.Size([32, 67, 5, 64])
        #

        #states = th.abs(self.state_transform(states.reshape(-1, self.state_dim)))

        # for V(s)
        v = self.calc_v(agent_qs)
        print("s: {}, agt_out: {}, obs: {}, v: {}".format(states.shape, agent_qs.shape, obs.shape, v.shape))
        #states = self.state_transform(states.reshape(-1, self.state_dim))
        #agent_qs = self.aqs_transform(agent_qs.view(-1, self.n_agents))
        #obs = self.obs_transform(obs.contiguous().view(-1, self.obs_dim))
        
        states = states.reshape(-1, self.state_dim)
        agent_qs = self.aqs_transform(agent_qs.view(-1, self.n_agents))
        obs = obs.contiguous().view(-1, self.obs_dim)
        #x = th.cat([states, agent_qs, obs], dim=1)
        g_states = self.state_transform(th.cat([states, obs], dim=1))
        x = th.cat([g_states, agent_qs], dim=1)
        #print("s: {}, agt_out: {}, obs: {}, v: {}, x: {}".format(states.shape, agent_qs.shape, obs.shape, v.shape, x.shape))

        # for combined experiments
        #states = states.reshape(-1, self.state_dim)
        #agent_qs = agent_qs.view(-1, self.n_agents)
        #histories = histories.contiguous().view(-1, self.hist_dim)

        #x = th.cat([states, agent_qs, histories], dim=1)

        #print("s: {}, agt_out: {}, obs: {}, v: {}".format(states.shape, agent_qs.shape, obs.shape, v.shape))
        #
        # s: torch.Size([2144, 120]), agt_out: torch.Size([2144, 5]), hist: torch.Size([2144, 320])
        #
        #print("---")

        #states = rearrange(states, 'a d -> d a')
        #agent_qs = rearrange(agent_qs, 'd -> () d')
        #histories = rearrange(histories, 'd -> () d')
        #print("s: {}, agt_out: {}, hist: {}".format(states.shape, agent_qs.shape, histories.shape))
        #exit(0)
        #q = self.mixer(g_states, agent_qs) #self.mixer(states, agent_qs, obs)

        # for combined experiments
        q = self.mixer(x)

        #print('q: {}, v: {}'.format(q.shape, v.shape))
        #v = rearrange(v, '(b n) d -> b n d', b=self.args.batch_size)
        q_tot = q + v
        #states = rearrange(states, '(b n) d -> b n d', b=self.args.batch_size)
        #v = rearrange(v, '(b n) d -> b n d', b=self.args.batch_size)
        q_tot = rearrange(q_tot, '(b n) d -> b n d', b=self.args.batch_size)
        #print('q_tot: ', q_tot.shape)
        #b, n = q_tot.shape[0], q_tot.shape[1]
        #q_tot = q_tot.max(dim=-1, keepdim=True)[0]

        #print('q_tot: ', q_tot.shape)
        return q_tot


        
