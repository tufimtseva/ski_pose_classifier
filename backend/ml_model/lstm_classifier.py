import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, batch_size, device=torch.device("cpu")):
        super(LSTMClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        self.lstm = nn.LSTM(input_size= self.input_dim ,hidden_size=self.hidden_dim,
                           num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        hidden = self.init_hidden(self.batch_size)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)

        return out
    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.num_layers,batch_size,self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.num_layers,batch_size,self.hidden_dim)).to(self.device)
        hidden = (h0,c0)
        return hidden