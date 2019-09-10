import torch
import torch.nn as nn
import torch.nn.functional as F

class ESA(nn.Module):
    def __init__(self, pred2ix_size, pred_embedding_dim, transE_dim, hidden_size, method, device):
        super(ESA, self).__init__()
        self.pred2ix_size = pred2ix_size
        self.pred_embedding_dim = pred_embedding_dim
        self.transE_dim = transE_dim
        self.input_size = self.transE_dim + self.pred_embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.pred2ix_size, self.pred_embedding_dim)
        self.lstm_feature = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True)
        self.lstm_user = nn.LSTM(6, 6, bidirectional=True)
        self.attn_1 = Attn(method, 2*hidden_size)
        self.attn_2 = Attn(method, 2*hidden_size)
        self.attn_3 = Attn(method, 2*hidden_size)
        self.attn_4 = Attn(method, 2*hidden_size)
        self.attn_5 = Attn(method, 2*hidden_size)
        self.attn_6 = Attn(method, 2*hidden_size)
        self.linear = nn.Linear(12, 6)
        self.hidden = self._init_hidden()
        self.device = device

    def forward(self, input_tensor):
        # input representation
        pred_embedded = self.embedding(input_tensor[0])
        obj_embedded = input_tensor[1]
        embedded = torch.cat((pred_embedded, obj_embedded), 2)

        # feature extraction
        lstm_out, (hidden_state, cell_state) = self.lstm_feature(embedded, self.hidden)

        # entity-phase: mulit-aspect attention
        atten_weight_1 = self.attn_1(lstm_out, hidden_state)
        atten_weight_2 = self.attn_2(lstm_out, hidden_state)
        atten_weight_3 = self.attn_3(lstm_out, hidden_state)
        atten_weight_4 = self.attn_4(lstm_out, hidden_state)
        atten_weight_5 = self.attn_5(lstm_out, hidden_state)
        atten_weight_6 = self.attn_6(lstm_out, hidden_state)
        atten_weight_all = torch.cat((
            atten_weight_1,
            atten_weight_2,
            atten_weight_3,
            atten_weight_4,
            atten_weight_5,
            atten_weight_6),0).t()

        # user-phase attention: bi-lstm + attention
        user_weight_all = atten_weight_all.unsqueeze(1)
        lstm_out, (hidden_state, cell_state) = self.lstm_user(user_weight_all)
        hidden_state = self.linear(hidden_state.view(-1))
        user_weight = hidden_state.view(-1, 1)
        atten_weight = torch.mm(atten_weight_all, user_weight).t()
        atten_weight = F.softmax(atten_weight, dim=1)
        return atten_weight

    def _init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size, device=self.device),
            torch.randn(2, 1, self.hidden_size, device=self.device))

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ["dot", "general", "original"]:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == "general" or self.method == "original":
            self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, lstm_out, hidden_state):
        hidden_state = hidden_state.view(1, 1, -1)
        # methods
        if self.method == "dot":
            atten_weight = self._dot(lstm_out, hidden_state)
        elif self.method == "general":
            atten_weight = self._general(lstm_out, hidden_state)
        elif self.method == "original":
            atten_weight = self._original(lstm_out, hidden_state)

        return atten_weight.t()

    def _dot(self, lstm_out, hidden_state):
        atten_weight = torch.sum(hidden_state * lstm_out, dim=2)
        return atten_weight

    def _general(self, lstm_out, hidden_state):
        hidden_state = self.linear(hidden_state)
        atten_weight = torch.sum(hidden_state * lstm_out, dim=2)
        return atten_weight

    def _original(self, lstm_out, hidden_state):
        hidden_state = self.linear(hidden_state).view(1, -1, 1)
        lstm_out = lstm_out.permute(1, 0, 2)
        atten_weight = torch.bmm(lstm_out, hidden_state).squeeze(2)
        return atten_weight.t()