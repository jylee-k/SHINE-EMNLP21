import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(DenseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.a.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # (N, out_features)

        # Attention mechanism
        N = Wh.size(0)
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1)  # (N*N, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a)).view(N, N)

        # Masked attention (only for edges that exist)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # (N, out_features)

        return h_prime

class SimpleDenseGAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleDenseGAT, self).__init__()
        self.gat1 = DenseGATLayer(in_features, hidden_features)
        self.gat2 = DenseGATLayer(hidden_features, out_features)

    def forward(self, x, adj):
        x = F.elu(self.gat1(x, adj))
        x = self.gat2(x, adj)
        return x

# Example usage:
# x: (N_nodes, in_features)
# adj: (N_nodes, N_nodes) adjacency matrix (0/1 or weighted)

# model = SimpleDenseGAT(in_features=x.shape[1], hidden_features=8, out_features=2)
# out = model(x, adj)
