import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.RNN import LSTM
from speechbrain.lobes.models.RNNLM import RNNLM
from speechbrain.nnet.transducer.transducer_joint import Transducer_joint

class DeepRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blank_index = config.blank_index
        self.vocab_size = config.vocab_size
        self.encoder = LSTM(input_size=config.feature_size, hidden_size=config.hidden_size, num_layers=config.num_layers, bidirectional=config.bidirectional, dropout=config.dropout)
        self.outnet = Linear(input_size=config.hidden_size * 2, n_neurons=self.vocab_size)
    
    def forward(self, x):
        y, h = self.encoder(x)
        return self.outnet(y), h # [N, T, V]
        

class Transducer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.blank_index = config.blank_index
        self.encoder = LSTM(input_size=config.feature_size, hidden_size=config.hidden_size, num_layers=config.num_layers, bidirectional=config.bidirectional)
        self.encoder_out = Linear(input_size=config.hidden_size * 2, n_neurons=config.hidden_size)
        self.decoder = RNNLM(rnn_layers=1, return_hidden=True, embedding_dim=config.vocab_size-1,
                             rnn_neurons=config.hidden_size, dnn_neurons=config.hidden_size, output_neurons=config.vocab_size)
        self.decoder.out = Linear(input_size=config.hidden_size, n_neurons=config.hidden_size)
        joint_net = Linear(input_size=config.hidden_size * 2, n_neurons=config.hidden_size)
        self.jointer = Transducer_joint(joint_net, joint='concat', nonlinearity=nn.Tanh)
        self.out_net = Linear(input_size=config.hidden_size, n_neurons=config.vocab_size)
    
    def encode(self, x):
        encode_state, _ = self.encoder(x)
        return self.encoder_out(encode_state)
        
    def forward(self, x, y):
        encode_state = self.encode(x)
        decode_state, _ = self.decoder(F.pad(y, pad=[1,0,0,0], value=self.blank_index))
        logits = self.jointer(encode_state.unsqueeze(2), decode_state.unsqueeze(1))
        return F.log_softmax(self.out_net(logits), dim=-1) # [N, T, U, V]