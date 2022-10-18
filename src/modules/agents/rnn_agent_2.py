import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent2(nn.Module):
    def __init__(self, input_shape, args, scheme):
        super(RNNAgent2, self).__init__()
        self.args = args
        self.scheme = scheme
        self.input_shape = input_shape
        self.in_channels = scheme["obs"]["vshape"] #self.args.batch_size_run * self.args.n_agents
        self.out_channels = self.args.out_channels
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        #self.hidden = None
        self.conv1x = nn.Conv1d(input_shape, self.out_channels, kernel_size=1)
        self.conv3x = nn.Conv1d(input_shape, self.out_channels, kernel_size=3, padding=1)        
        self.conv5x = nn.Conv1d(input_shape, self.out_channels, kernel_size=5, padding=2)
        self.mish = nn.Mish()
        self.max_pool = nn.MaxPool1d(2)
        self.rnn = nn.GRUCell(args.out_channels // 2, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        #return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        nn.init.kaiming_normal_(self.conv1x.weight)
        
        #self.hidden = torch.rand(1, self.args.rnn_hidden_dim).to("cuda")
        #nn.init.kaiming_normal_(self.hidden)
        #nn.init.kaiming_normal_(self.fc1.weight) #(1, self.args.rnn_hidden_dim).zero_()
        return nn.init.kaiming_normal_(self.fc1.weight.new(1, self.args.rnn_hidden_dim))
        
        #return self.conv1x

    def forward(self, inputs, hidden_state):
        #print("inputs: {}, inp_shape: {}, args.n_agent: {}".format(inputs.shape, self.input_shape, self.args.n_agents)) # inputs = [bs * n_agents, obs]
        # exit(0)
        inputs = inputs.unsqueeze(0).permute(0, 2, 1)
        #x = F.relu(self.fc1(inputs))
        x1 = self.mish(self.conv1x(inputs))
        #print("x1: {}".format(x1.shape))
        x2 = self.mish(self.conv3x(inputs))
        #print("x2: {}".format(x2.shape))
        x3 = self.mish(self.conv5x(inputs))
        #print("x3: {}".format(x3.shape))
        x = (x1 + x2 + x3).permute(0, 2, 1)
        x = self.max_pool(x).squeeze(0)
        #print("x: {}".format(x.shape))
        #print("hidd: {}".format(hidden_state.shape))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        #print("inputs: {}, h: {}, q: {}".format(inputs.shape, h.shape, q.shape)) # inputs = [bs * n_agents, obs]
        #exit(0)
        return q, h
