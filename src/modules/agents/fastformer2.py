# https://github.com/lucidrains/fast-transformer-pytorch/blob/main/fast_transformer_pytorch/fast_transformer_pytorch.py

import torch
from torch._C import device
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from torchsummary import summary

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# helper classes

class PreNorm(nn.Module):
    def __init__(self, input_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fn = fn

    def forward(self, states, agents_qs, obs, mask, **kwargs):
        s = self.norm(states)
        adv = self.norm(agents_qs)
        v = self.norm(obs)
        return self.fn(states, agents_qs, obs, mask, **kwargs)

# blocks

def FeedForward(input_dim, mult = 4):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * mult),
        nn.GELU(),
        nn.Linear(input_dim * mult, input_dim)
    )

class FastAttention(nn.Module):
    def __init__(
        self,
        dim,                
        *,
        heads = 8,
        dim_head = 64,
        max_seq_len = None,
        pos_emb = None,         
        n_agents = None,     
        args = None   
    ):
        super().__init__()
        self.dim = dim
        self.inner_dim = heads * dim_head
        self.dim_head = dim_head
        #self.q_dim = q_dim
        #self.kv_dim = kv_dim
        #self.v_dim = v_dim        
        self.heads = heads
        self.scale = dim_head ** -0.5
        #self.obs_dim = kv_dim
        #self.action_dim = q_dim #scheme['avail_actions']['vshape']
        self.n_agents = n_agents
        #self.input_dim = input_dim
        self.abs = abs
        self.args = args
        self.b_max = 0
        #self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False)
        self.q = nn.Linear(self.dim, self.inner_dim)
        self.k = nn.Linear(self.dim, self.inner_dim)
        self.v = nn.Linear(self.dim, self.inner_dim)

        

        # rotary positional embedding

        assert not (exists(pos_emb) and not exists(max_seq_len)), 'max_seq_len must be passed in if to use rotary positional embeddings'

        self.pos_emb = pos_emb
        self.max_seq_len = max_seq_len

        # if using relative positional encoding, make sure to reduce pairs of consecutive feature dimension before doing projection to attention logits

        kv_attn_proj_divisor = 1 if not exists(pos_emb) else 2

        self.to_q_attn_logits = nn.Linear(self.dim_head, 1, bias = False)  # for projecting queries to query attention logits
        #self.to_k_attn_logits = nn.Linear(self.dim_head, 1, bias = False)  # for projecting keys to key attention logits

        self.to_k_attn_logits = nn.Linear(self.dim_head // kv_attn_proj_divisor, 1, bias = False)

        # final transformation of values to "r" as in the paper

        #self.to_r = nn.Linear(self.dim_head, self.dim_head)
        self.to_r = nn.Linear(dim_head // kv_attn_proj_divisor, dim_head)

        self.to_out = nn.Linear(self.inner_dim, self.dim)

    def forward(self, states, agent_qs, obs, mask = None):
        # x = [bs * n_agents, input_shape]
        n = self.args.eps_limit
        device, h, use_rotary_emb = states.device, self.heads, exists(self.pos_emb)
        b = 1 if states.shape[0] < self.args.batch_size else self.args.batch_size
        #print('ff  x: {}'.format(x.shape))
        #n = x.shape[0]
        #device = x.device
        #h = self.heads
        #use_rotary_emb = exists(self.pos_emb)

        #qkv = self.to_qkv(x).chunk(3, dim = -1)
        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        #q = self.q(s)
        #k = self.k(adv)
        #v = self.v(v)

        q = torch.abs(self.q(states))
        k = self.k(agent_qs)
        v = self.v(obs)

        """ q = self.q(hist)
        k = self.k(s)
        v = self.v(agent_qs) """

        """ q = self.q(hist)
        k = self.k(agent_qs)
        v = self.v(s) """

        """ q = self.q(agent_qs)
        k = self.k(hist)
        v = torch.abs(self.v(s)) """
        
        """ q = rearrange(q, '(b n) (h d) -> b h n d', b=self.args.batch_size, h=h)
        k = rearrange(k, '(b n) (h d) -> b h n d', b=self.args.batch_size, h=h)
        v = rearrange(v, '(b n) (h d) -> b h n d', b=self.args.batch_size, h=h) """

        if(len(q.shape)==2):
            q = rearrange(q, '(b n) (h d) -> b h n d', b=b, h=h)
            k = rearrange(k, '(b n) (h d) -> b h n d', b=b, h=h)
            v = rearrange(v, '(b n) (h d) -> b h n d', b=b, h=h)
        elif(len(q.shape)==3):
            q = rearrange(q, 'b n (h d) -> b h n d', b=b, h=h)
            k = rearrange(k, 'b n (h d) -> b h n d', b=b, h=h)
            v = rearrange(v, 'b n (h d) -> b h n d', b=b, h=h)

        #q = rearrange(q, 'b (h d) -> b h d', h=h)
        #k = rearrange(k, 'b (h d) -> b h d', h=h)
        #v = rearrange(v, 'b (h d) -> b h d', h=h)
        mask = torch.ones(1, q.shape[2]).bool().cuda()
        mask_value = -torch.finfo(q.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        #print('q: {}, k: {}, v: {}, m: {}'.format(q.shape, k.shape, v.shape, mask.shape))
       
        if use_rotary_emb:
            freqs = self.pos_emb(torch.arange(self.max_seq_len, device = device), cache_key = self.max_seq_len)
            #print('freqs: {}, b_max: {}'.format(freqs.shape, self.b_max))
            freqs = rearrange(freqs[:q.shape[2]], 'n d -> () () n d')
            #print('freqs: {}'.format(freqs.shape))
            q_aggr, k_aggr, v_aggr = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))
        else:
            q_aggr, k_aggr, v_aggr = q, k, v

        # calculate query attention logits

        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        #q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n -> b h n') * self.scale
        q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = q_attn_logits.softmax(dim = -1)
        #print('q_attn: {}, q_aggr: {}'.format(q_attn.shape, q_aggr.shape))
        # calculate global query token

        global_q = einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token

        k = k * global_q

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            k = reduce(k, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # now calculate key attention logits

        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = k_attn_logits.softmax(dim = -1)
        #print('k_attn: {}, k_aggr: {}'.format(k_attn.shape, k_aggr.shape))
        # calculate global key token

        global_k = einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        #global_k = rearrange(global_k, 'b h d -> b h d')
        global_k = rearrange(global_k, 'b h d -> b h () d')


        # bias the values

        u = v_aggr * global_k

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            u = reduce(u, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # transformation step

        r = self.to_r(u)

        # paper then says to add the queries as a residual

        r = r + q

        # combine heads

        #r = rearrange(r, 'b h d -> b (h d)')
        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)

# main class

class FastTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        absolute_pos_emb = False,             
        n_agents = None,
        out_dim = 0,
        abs = False,
        args = None
    ):
        super().__init__()
        #self.token_emb = nn.Embedding(num_tokens, dim)

        # positional embeddings
        self.b_max = 0
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if absolute_pos_emb else None

        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (dim_head % 4) == 0, 'dimension of the head must be divisible by 4 to use rotary embeddings'
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        # layers

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = FastAttention(dim, dim_head = dim_head, heads = heads, 
                                 pos_emb = layer_pos_emb, 
                                 max_seq_len = max_seq_len,
                                 n_agents=n_agents, args=args)
            ff = FeedForward(dim, mult = ff_mult)

            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                ff #PreNorm(dim, ff)
            ]))

        # weight tie projections across all layers

        first_block, _ = self.layers[0]
        for block, _ in self.layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)#args.avail_actions)
        )
        

    def forward(
        self,
        states,
        agent_outs,
        obs,
        mask = None,
        b_max = 0    
    ):
        n, device = agent_outs.shape[1], agent_outs.device
        #x = self.token_emb(x)
        self.b_max = b_max
        """ if exists(self.abs_pos_emb):            
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            pos_emb = rearrange(pos_emb, 'n d -> () n d')
            #print('FF  x: {}, pos_emb: {}'.format(x.shape, pos_emb.shape))
            states = states + pos_emb
            agent_outs = agent_outs + pos_emb
            obs = obs + pos_emb """
        #print(" FF s: {}, agt_out: {}, hist: {}, b_max: {}".format(states.shape, agent_outs.shape, histories.shape, self.b_max))
        #print('----')
        for attn, ff in self.layers:
            attn.b_max = self.b_max
            x = attn(states, agent_outs, obs , mask) 
            x = ff(x) + x

        return self.to_logits(x)

""" #t = torch.rand(1, 1920).cuda()#(32, 41, 51).view(-1, 51).cuda()
s = torch.rand(3840, 256)
a = torch.rand(3840, 256)
h = torch.rand(3840, 256)
mask = torch.ones(1, 256).bool().cuda()
#ff = FastTransformer(num_tokens = 96, dim = 1920, depth = 2, max_seq_len=2000, 
#                    absolute_pos_emb=False, n_agents=3, out_dim=11).cuda() #FastAttention(96, 80, 11, 11)

ff = FastTransformer(num_tokens = 120, dim = 256,
                                     depth = 4, max_seq_len = 120,
                                     absolute_pos_emb=False,
                                     out_dim=1, n_agents=5,
                                     args=self.args)

#summary(ff, (64, 96), batch_size=32, device='cpu')
t = ff.forward(t, t, t, mask)
print('t: {}'.format(t.shape))
t = t.view(32, -1, 11)
print('t: {}'.format(t.shape)) """