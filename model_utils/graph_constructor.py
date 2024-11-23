from torch import nn
import torch
import torch.nn.functional as F


class dynamic_graph_constructor(nn.Module):
    def __init__(self, feature_dim, k, hidden_dim, alpha=3):
        super(dynamic_graph_constructor, self).__init__()
        self.feature_dim = feature_dim
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate_static = nn.Linear(hidden_dim, hidden_dim)
        self.gate_dynamic = nn.Linear(hidden_dim, hidden_dim)

        self.emb1 = nn.Embedding(feature_dim, hidden_dim)
        self.emb2 = nn.Embedding(feature_dim, hidden_dim)

        self.idx = torch.arange(feature_dim)
        self.k = k
        self.hidden_dim = hidden_dim
        self.alpha = alpha

    def sparseA(self, emb):
        # generate sparsified adjacency matrix
        static_nodevec1 = self.emb1(self.idx)  # N, H
        static_nodevec2 = self.emb2(self.idx)
        nodevec2 = nodevec1 = emb
        nodevec1_gate = torch.sigmoid(self.gate_static(static_nodevec1) + self.gate_dynamic(nodevec1))
        nodevec1 = (1 - nodevec1_gate) * nodevec1 + nodevec1_gate * static_nodevec1
        nodevec2_gate = torch.sigmoid(self.gate_static(static_nodevec2) + self.gate_dynamic(nodevec2))
        nodevec2 = (1 - nodevec2_gate) * nodevec2 + nodevec2_gate * static_nodevec2

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.matmul(nodevec1, nodevec2.transpose(1, 2)) - torch.matmul(nodevec2, nodevec1.transpose(1, 2))
        adj = F.relu(torch.tanh(self.alpha * a))  # B, N, N
        mask = torch.zeros(emb.size(0), self.idx.size(0), self.idx.size(0)).to(self.device)
        mask.fill_(float("0"))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 2)
        mask.scatter_(2, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def forward(self, emb):
        # generate full adjacency matrix
        self.idx = self.idx.to(emb.device)
        static_nodevec1 = self.emb1(self.idx)  # N, H
        static_nodevec2 = self.emb2(self.idx)
        nodevec2 = nodevec1 = emb
        nodevec1_gate = torch.sigmoid(self.gate_static(static_nodevec1) + self.gate_dynamic(nodevec1))
        nodevec1 = (1 - nodevec1_gate) * nodevec1 + nodevec1_gate * static_nodevec1
        nodevec2_gate = torch.sigmoid(self.gate_static(static_nodevec2) + self.gate_dynamic(nodevec2))
        nodevec2 = (1 - nodevec2_gate) * nodevec2 + nodevec2_gate * static_nodevec2

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.matmul(nodevec1, nodevec2.transpose(1, 2)) - torch.matmul(nodevec2, nodevec1.transpose(1, 2))
        adj = F.relu(torch.tanh(self.alpha * a))  # B, N, N
        return adj


def dyna_sparse_graph(adj, idx, k):
    mask = torch.zeros(adj.size(0), idx.size(0), idx.size(0)).to(adj.device)
    mask.fill_(float("0"))
    s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(k, 2)
    mask.scatter_(2, t1, s1.fill_(1))
    adj = adj * mask
    return adj


def construct_cosine_graph(state, topK):

    embeddings = F.normalize(state, dim=-1, p=2)
    dynamic_graph = torch.bmm(embeddings, embeddings.transpose(1, 2))

    # only retain the top k edges
    mask = torch.zeros(dynamic_graph.size(0), dynamic_graph.size(1), dynamic_graph.size(2)).to(dynamic_graph.device)
    mask.fill_(float("0"))
    s1, t1 = (dynamic_graph + torch.rand_like(dynamic_graph) * 0.01).topk(topK, 2)
    mask.scatter_(2, t1, s1.fill_(1))
    dynamic_graph = dynamic_graph * mask

    return dynamic_graph
