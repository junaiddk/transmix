REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_agent_2 import RNNAgent2
from .transformer_agent import TransformerAgent
from .transformer_agent_2 import TAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn2"] = RNNAgent2
REGISTRY["t_agent"] = TransformerAgent
REGISTRY["tt_agent"] = TAgent