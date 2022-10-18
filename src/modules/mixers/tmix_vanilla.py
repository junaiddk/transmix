
from numpy.core.fromnumeric import shape
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class TMixer(nn.Module):
    def __init__(self, scheme, input_shape, args):
        super(TMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.scheme = scheme
        
        self.input_shape = input_shape
        self.state_dim = int(np.prod(args.state_shape))
        self.n_actions = args.n_actions
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.max_seq_len = args.batch_size * args.eps_limit
        self.embed_dim = args.embed_dim
        self.obs_dim = self.n_agents * self.scheme['obs']['vshape']
        self.hist_dim = self.args.n_agents * self.args.rnn_hidden_dim
        dim = args.embed_dim * 3
        
        self.state_transform = nn.Linear(self.state_dim, args.embed_dim)        
        self.aqs_transform = nn.Linear(self.n_agents, args.embed_dim)        
        self.hist_transform = nn.Linear(self.hist_dim, args.embed_dim)            
        
        self.enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=args.heads, dim_feedforward=args.ff, dropout=0.4, 
                                                    activation=F.relu, batch_first=True, device='cuda')
        self.mixer = nn.TransformerEncoder(self.enc_layer, num_layers=args.t_depth)
        self.bottleneck = nn.Linear(dim, dim)
        self.out = nn.Sequential( 
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1)
            
        )
    def create_noise(self, states, mean=0, stddev=0.05):
        noise = th.as_tensor(states, dtype=th.float).normal_(mean, stddev).cuda()
        return noise
    
    def calc_v(self, agent_qs):
        v_tot = th.sum(agent_qs, dim=-1, keepdim=True)
        return v_tot

    def forward(self, agent_qs, hist, states, b_max=0):  
        v = self.calc_v(agent_qs)

        if self.args.is_noise == True:
            noise = self.create_noise(states)
            states = ((noise + states).detach() - states).detach() + states

        states = th.abs(self.state_transform(states))
        agent_qs = self.aqs_transform(agent_qs) 
        
        hist = hist.contiguous().view(hist.shape[0], hist.shape[1], -1)
        
        hist = self.hist_transform(hist)
        
        
        # This one is based on the vanilla transformer 
        x = th.cat([states, hist, agent_qs], dim=2)
        x = self.bottleneck(x)
        q = self.mixer(x) + x
        q_tot = (self.out(q) * x) + v
        del x, q, v
        
        return q_tot


        
