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

class NetConv_Swi(torch.nn.Module):
    def __init__(self, in_nf, in_ef, out_nf, h1=8, h2=16, h_in=24, h_ma=4):
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
        self.net_in_trans1 = MLP(self.in_nf, h_in,h_in,h_in, self.ma * self.ma)
        self.edge_matrix_transform1 = MLP(
            self.in_nf + self.in_ef, h_in,h_in,h_in, self.ma * self.ma)

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

    def forward(self, g, nf):
        with g.local_scope():
            g.ndata['nf'] = nf.to(torch.float32)
            # input direction
            g.update_all(self.edge_msg_i, fn.sum('efi', 'new_nf'), etype='net_out')
            # output direction
            _, dst_nodes = g.edges(etype='net_in')
            g.apply_nodes(self.dri_trans1, dst_nodes)
            g.apply_edges(self.raw_attention1, etype='net_in')
            g.update_all(fn.copy_e('raw_attention1', 'raw_attention1'), fn.sum(
                'raw_attention1', 'attention_sum1'), etype='net_in')
            # message passing from sink to source
            g.apply_edges(self.edge_msg_o, etype='net_in')
            g.update_all(fn.copy_e('efo1', 'efo1'), fn.sum(
                'efo1', 'nfo1'), etype='net_in')
            g.update_all(fn.copy_e('efo2', 'efo2'), fn.sum(
                'efo2', 'nfo2'), etype='net_in')
            g.apply_nodes(self.node_reduce_o, dst_nodes)

            return g.ndata['new_nf'] 


class NetConv(torch.nn.Module):
    def __init__(self, in_nf, in_ef, out_nf, h1=8, h2=8, h_in=16):
        super().__init__()
        self.in_nf = in_nf
        self.in_ef = in_ef
        self.out_nf = out_nf
        self.h1 = h1
        self.h2 = h2
        self.MLP_msg_o2i = MLP(self.in_nf * 2 + self.in_ef,
                               h_in, self.out_nf)
        self.MLP_reduce_o = MLP(self.in_nf + self.h1 +
                                self.h2, h_in, self.out_nf)
        self.MLP_msg_i2o = MLP(self.in_nf * 2 + self.in_ef,
                               h_in, 1 + self.h1 + self.h2)

    def edge_msg_i(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf'],
                      edges.data['nef']], dim=1)
        x = self.MLP_msg_o2i(x)
        return {'efi': x}

    def edge_msg_o(self, edges):
        x = torch.cat([edges.src['nf'], edges.dst['nf'],
                      edges.data['nef']], dim=1)
        x = self.MLP_msg_i2o(x)
        k, f1, f2 = torch.split(x, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'efo1': f1 * k, 'efo2': f2 * k}

    def node_reduce_o(self, nodes):
        x = torch.cat([nodes.data['new_nf'], nodes.data['nfo1'],
                      nodes.data['nfo2']], dim=1)
        x = self.MLP_reduce_o(x)
        return {'new_nf': x}

    def forward(self, g, nf):
        with g.local_scope():
            g.ndata['new_nf'] = g.ndata['nf'] = nf.to(torch.float32)
            # input direction
            g.update_all(self.edge_msg_i, fn.sum('efi', 'new_nf'), etype='net_out')
            # output direction
            g.apply_edges(self.edge_msg_o, etype='net_in')
            g.update_all(fn.copy_e('efo1', 'efo1'), fn.sum(
                'efo1', 'nfo1'), etype='net_in')
            g.update_all(fn.copy_e('efo2', 'efo2'), fn.mean(
                'efo2', 'nfo2'), etype='net_in')
            _, dst_nodes = g.edges(etype='net_in')
            g.apply_nodes(self.node_reduce_o, dst_nodes)

            return g.ndata['new_nf']

class CellConv(torch.nn.Module):
    def __init__(self, in_nf, in_cell_num_luts, in_cell_lut_sz, out_nf, out_cef, h1=8, h2=8, h_in=16, lut_dup=2):
        super().__init__()
        self.in_nf = in_nf
        self.in_cell_num_luts = in_cell_num_luts
        self.in_cell_lut_sz = in_cell_lut_sz
        self.out_nf = out_nf
        self.out_cef = out_cef
        self.h1 = h1
        self.h2 = h2
        self.lut_dup = lut_dup

        self.MLP_lut_query = MLP(
            self.out_nf + 2 * self.in_nf, h_in, h_in, self.in_cell_num_luts * lut_dup * 2 + self.in_cell_num_luts * lut_dup)
        self.MLP_lut_attention = MLP(
            1 + 2 + self.in_cell_lut_sz * 2, h_in, h_in, self.in_cell_lut_sz * 2)
        self.MLP_lut_attention_input = MLP(
            1 + 1 + self.in_cell_lut_sz, h_in, h_in, self.in_cell_lut_sz)
        self.MLP_cellarc_msg = MLP(self.in_cell_num_luts *
                                   self.lut_dup, 2)
        self.MLP_for_scale = MLP(self.in_nf + 2 * self.in_nf + self.lut_dup + self.lut_dup, h_in, h_in, 2)
        
    def edge_msg_cell(self, edges):
        # generate lut axis query
        last_nf = edges.src['new_nf']

        q = torch.cat([last_nf, edges.src['nf'], edges.dst['nf']], dim=1)
        q = self.MLP_lut_query(q)
        q = q.reshape(-1, 3) # two for cell arc one for input
        q, q_input = torch.split(q,[2,1],dim=1)

        # answer lut axis query
        axis_len = self.in_cell_num_luts * (1 + 2 * self.in_cell_lut_sz)
        axis_len_input = self.in_cell_num_luts * (1 + self.in_cell_lut_sz)
        axis = edges.data['ef'][:, :axis_len]
        axis_input = edges.data['ef_input'][:, :axis_len_input]
        axis = axis.reshape(-1, 1 + 2 * self.in_cell_lut_sz)
        axis = axis.repeat(1, self.lut_dup).reshape(-1,
                                                    1 + 2 * self.in_cell_lut_sz)
        axis_input = axis_input.reshape(-1, 1 + self.in_cell_lut_sz)
        axis_input = axis_input.repeat(1, self.lut_dup).reshape(-1,
                                                    1 + self.in_cell_lut_sz)
        a = self.MLP_lut_attention(torch.cat([q, axis], dim=1))
        a_input = self.MLP_lut_attention_input(torch.cat([q_input, axis_input], dim=1))
        torch.set_printoptions(precision=10)

        # transform answer to answer mask matrix
        a = a.reshape(-1, 2, self.in_cell_lut_sz)
        ax, ay = torch.split(a, [1, 1], dim=1)
        a_input = a_input.reshape(-1, 1, self.in_cell_lut_sz)
        a = torch.matmul(ax.reshape(-1, self.in_cell_lut_sz, 1),
                         ay.reshape(-1, 1, self.in_cell_lut_sz))  # batch tensor product
        a_sigmoid = torch.sigmoid(a)
        a_sigmoid_input = torch.sigmoid(a_input)

        # Normalizing a_sigmoid such that the sum of all elements is 1
        a = a_sigmoid / (torch.sum(a_sigmoid, dim=[1, 2], keepdim=True) + 1e-10)
        a_input = a_sigmoid_input / (torch.sum(a_sigmoid_input, dim=[1, 2], keepdim=True) + 1e-10)
        # look up answer matrix in lut
        tables_len = self.in_cell_num_luts * self.in_cell_lut_sz ** 2
        tables = edges.data['ef'][:, axis_len:axis_len + tables_len]

        tables_len_input = self.in_cell_num_luts * self.in_cell_lut_sz
        tables_input = edges.data['ef_input'][:, axis_len_input:axis_len_input + tables_len_input]
        r = torch.matmul(tables.reshape(-1, 1, 1, self.in_cell_lut_sz ** 2),
                         a.reshape(-1, self.lut_dup, self.in_cell_lut_sz ** 2, 1))   # batch dot product
        r_input = torch.matmul(tables_input.reshape(-1,1,1,self.in_cell_lut_sz),
                               a_input.reshape(-1, self.lut_dup,self.in_cell_lut_sz, 1))

        # construct final msg
        r = r.reshape(len(edges), self.in_cell_num_luts * self.lut_dup)
        r_input = r_input.reshape(len(edges), self.in_cell_num_luts * self.lut_dup)
        x = torch.cat([last_nf, edges.src['nf'], edges.dst['nf'], r, r_input], dim=1)
        x = self.MLP_for_scale(x)
        scale, scale_input = torch.split(x,[1,1],dim=1)
        scale = torch.sigmoid(scale)
        scale_input = torch.sigmoid(scale_input)
        cef = torch.mean(r,dim=1) * (0.9 + scale.squeeze(1)/5)
        cef_input = torch.mean(r_input,dim=1) * (0.9 + scale_input.squeeze(1)/5)
        return {'efce': cef, 'efce_input': cef_input}
        
    def forward(self, g, nin, nf):
        # Get cell arc
        with g.local_scope():
            g.ndata['new_nf'] = nf.to(torch.float32)
            g.ndata['nf'] = nin.to(torch.float32)
            g.apply_edges(self.edge_msg_cell, etype='cell_out')
        
            return g.edges['cell_out'].data['efce'], g.edges['cell_out'].data['efce_input']

class Prediction(torch.nn.Module):
    def __init__(self, ndim_driver):
        super().__init__()
        self.embedding = MLP(ndim_driver,24, 12, 4) # For driver one-hot process
        self.nc1 = NetConv(24, 13, 24)
        self.nc2 = NetConv_Swi(24, 13, 24)
        self.nc3 = NetConv_Swi(24, 13, 24)
        self.cellarc = CellConv(24,1,8,24,1)
        self.dri_mlp = MLP(4, 2, 1)

    def forward(self, g, freq, util):
        em = g.ndata['one-hot-driver'].to(torch.float32)
        x2 = self.embedding(em)
        dri_sca = self.dri_mlp(x2).squeeze(1)

        # Data norm
        freq_tensor = torch.tensor(
            [(float(freq) - 350) / 150], dtype=torch.float32, device=g.device)
        util_tensor = torch.tensor(
            [float(util) - 57.5], dtype=torch.float32, device=g.device)

        freq_tensor = freq_tensor.view((-1, 1)).repeat(g.number_of_nodes(), 1)
        util_tensor = util_tensor.view((-1, 1)).repeat(g.number_of_nodes(), 1)

        nin = torch.cat(
            (x2, freq_tensor, util_tensor, g.ndata['nf']), dim=1)

        x_int = self.nc1(g, nin)
        e, e_input = self.cellarc(g, nin, x_int)
        x_swi = self.nc2(g, nin)
        x_swi = self.nc3(g, x_swi)
        x_swi = torch.mean(x_swi, dim=1) * (0.9 + torch.sigmoid(dri_sca)/5)

        return x_swi, e, e_input