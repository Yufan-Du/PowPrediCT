import torch
import torch.nn.functional as F
import dgl.function as fn
import dgl
from tqdm import tqdm

class MLP(torch.nn.Module):
    def __init__(self, *sizes, batchnorm=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                if batchnorm:
                    fcs.append(torch.nn.BatchNorm1d(sizes[i]))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)

class NetConv_Swi_cts(torch.nn.Module):
    def __init__(self, in_nf, in_ef, out_nf, h1=4, h2=8, h_in=16, h_ma=4):
        super().__init__()
        self.in_nf = in_nf
        self.in_ef = in_ef
        self.out_nf = out_nf
        self.h1 = h1
        self.h2 = h2
        self.ma = h_ma
        self.MLP_msg_o2i = MLP(self.in_nf + self.out_nf + self.in_ef,
                               h_in,h_in,h_in, self.out_nf)
        self.MLP_reduce_o = MLP(self.in_nf + self.h1 +
                                self.h2, h_in,h_in,h_in, self.out_nf)
        self.MLP_msg_i2o = MLP(self.in_nf + self.out_nf + self.in_ef,
                               h_in,h_in,h_in, 1 + self.h1 + self.h2)
        self.net_in_trans1 = MLP(self.in_nf, h_in, self.ma * self.ma)
        self.edge_matrix_transform1 = MLP(
            self.in_nf + self.in_ef, h_in, self.ma * self.ma)

    def edge_msg_i(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf'],
                      edges.data['nef']], dim=1)
        x = self.MLP_msg_o2i(x)
        return {'efi': x}

    def edge_msg_o(self, edges):
        attention_score1 = edges.data['raw_attention1'] / \
            (edges.dst['attention_sum1'] + 1e-3)
        
        x = torch.cat([edges.src['new_nf'], edges.dst['nf'],
                      edges.data['nef']], dim=1)
        x = self.MLP_msg_i2o(x)
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'efo1': f1 * k * attention_score1, 'efo2': f2 * k}

    def node_reduce_o(self, nodes):
        x = torch.cat([nodes.data['new_nf'], nodes.data['nfo1'],
                      nodes.data['nfo2']], dim=1)
        x = self.MLP_reduce_o(x)
        return {'new_nf': x}

    def dri_trans1(self, nodes):
        return {'node_matrix1': self.net_in_trans1(nodes.data['new_nf']).view(-1, self.ma, self.ma)}

    def raw_attention1(self, edges):
        node_matrix = edges.dst['node_matrix1']
        edge_matrix = self.edge_matrix_transform1(torch.cat(
            [edges.src['new_nf'], edges.data['nef']], dim=1)).view(-1, self.ma, self.ma)
        raw_attention = torch.sigmoid(torch.bmm(
            node_matrix, edge_matrix.transpose(1, 2)).sum(dim=(-1, -2)))
        return {'raw_attention1': raw_attention.unsqueeze(-1)}

    def forward(self, g, nf, dri_pin):
        g.ndata['nf'] = nf.to(torch.float32)
        
        # Message passing out from dri_pin nodes through 'net_out' edges
        A, B = g.out_edges(dri_pin, etype='net_out')
        g.send_and_recv((A, B), self.edge_msg_i, fn.sum('efi', 'new_nf'), etype='net_out')
        
        # Apply node and edge functions only on relevant nodes and edges linked to dri_pin via net_out
        g.apply_nodes(self.dri_trans1, B)
        
        # Get source nodes for the 'net_in' edges pointing to dri_pin
        src_nodes_in, B = g.in_edges(dri_pin, etype='net_in')
        g.apply_edges(self.raw_attention1, (src_nodes_in, B), etype='net_in')
        g.send_and_recv((src_nodes_in, B), fn.copy_e('raw_attention1', 'raw_attention1'), fn.sum('raw_attention1', 'attention_sum1'), etype='net_in')
        
        # Message passing from other nodes to dri_pin through 'net_in' edges
        g.apply_edges(self.edge_msg_o, (src_nodes_in, B), etype='net_in')
        g.send_and_recv((src_nodes_in, B), fn.copy_e('efo1', 'efo1'), fn.sum('efo1', 'nfo1'), etype='net_in')
        g.send_and_recv((src_nodes_in, B), fn.copy_e('efo2', 'efo2'), fn.sum('efo2', 'nfo2'), etype='net_in')
        
        # Apply node function only on dri_pin nodes
        g.apply_nodes(self.node_reduce_o, dri_pin)
        
        return g.ndata['new_nf']
class Prediction_cts(torch.nn.Module):
    def __init__(self, ndim_driver):
        super().__init__()
        self.embedding = MLP(ndim_driver,24, 12, 4)
        self.nc1 = NetConv_Swi_cts(24, 22, 24)
        self.nc2 = NetConv_Swi_cts(24, 22, 24)
        self.dri_mlp = MLP(4, 2, 1)

    def forward(self, g, freq, util, dri_pin):
        em = g.ndata['one-hot-driver'].to(torch.float32)
        x2 = self.embedding(em)
        dri_sca = self.dri_mlp(x2).squeeze(1)

        freq_tensor = torch.tensor(
            [(float(freq) - 350) / 150], dtype=torch.float32, device=g.device)
        util_tensor = torch.tensor(
            [float(util) - 57.5], dtype=torch.float32, device=g.device)

        freq_tensor = freq_tensor.view((-1, 1)).repeat(g.number_of_nodes(), 1)
        util_tensor = util_tensor.view((-1, 1)).repeat(g.number_of_nodes(), 1)

        nin = torch.cat(
            (x2, util_tensor, freq_tensor, g.ndata['nf']), dim=1)

        x_swi = self.nc1(g, nin, dri_pin)
        x_swi = self.nc2(g, x_swi, dri_pin)
        x_swi_out = x_swi.clone()
        x_swi = torch.mean(x_swi, dim=1) * (0.9 + torch.sigmoid(dri_sca)/5)
        return x_swi, x_swi_out