import torch
import torch.nn as nn
import dgl.function as fn
from config import *

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


class CNNEncoder(nn.Module):
    def __init__(self, out_f=8):
        super(CNNEncoder, self).__init__()
        self.out_f = out_f
        # Convolution 1
        self.conv1 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution 2
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution 3
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution 4
        self.conv4= nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution 5
        self.conv5= nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mlp = MLP(10 * 8 * 8, 128 , 64, self.out_f)

    def forward(self, x ,masks):
        x = x.to(torch.float32)
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.reshape(x.size(0), 64,64)
        masks_num = masks.shape[0]
        x = x.unsqueeze(0).expand(masks_num,-1,-1,-1)
        x = (x * masks).to(dtype=torch.float32)

        x = self.pool3(self.act3(self.conv3(x)))
        x = self.pool4(self.act4(self.conv4(x)))
        x = self.pool5(self.act5(self.conv5(x)))

        x = x.reshape(masks_num,-1)
        x = self.mlp(x)
        return x


class Prop_test(torch.nn.Module):
    def __init__(self, in_nf, e_nf, out_nf, h_in=8, h1=2, h2=6):
        super().__init__()
        self.in_nf = in_nf
        self.e_nf = e_nf
        self.out_nf = out_nf
        self.h_in = h_in
        self.h1 = h1
        self.h2 = h2
        self.net_prop = MLP(2 * self.in_nf, h_in, 1 + self.h1 + self.h2)
        self.reduce = MLP(in_nf + h1 + h2, h_in, out_nf)
        self.res = MLP(out_nf, 4, 2)
        self.layer = 0

    def edge_msg(self, edges):
        edge_info = torch.cat([edges.src['new_nf'], edges.dst['nf']], dim=1)
        edge_info = self.net_prop(edge_info)
        k, f1, f2 = torch.split(edge_info, [1, self.h1, self.h2], dim=1)
        k = torch.sigmoid(k)
        return {'ef1': k*f1, 'ef2': k*f2}

    def adding(self, edges):
        power_info = edges.src['pwr']
        power_info_feat = edges.src['pwr-feat']
        return {'epwr': power_info, 'epwr-feat': power_info_feat}

    def module_reduce(self, nodes):
        c = torch.cat([nodes.data['nf'], nodes.data['nf1'],
                      nodes.data['nf2']], dim=1)
        c = self.reduce(c)
        res = self.res(c)
        res[:,0] = res[:,0] / 100 / (self.layer**10)
        res[:,1] = res[:,1] / 100 / (self.layer**10)
        next_pwr = nodes.data['pwr'].clone()
        next_pwr[:,0] = nodes.data['pwr'][:,0] * (0.95 + torch.sigmoid(res[:,0])/10)
        next_pwr_feat = nodes.data['pwr-feat'].clone()
        next_pwr_feat[:,2] = nodes.data['pwr-feat'][:,2] * (0.95 + torch.sigmoid(res[:,1])/10)

        return {'pwr': next_pwr, 'new_nf': c, 'pwr-feat': next_pwr_feat}

    def forward(self, g, topo):
        self.layer = 0
        for i in range(1, len(topo)):
            self.layer = i
            layer_nodes = topo[i]
            es = g.in_edges(layer_nodes, etype='father')
            g.apply_edges(self.edge_msg, es, etype='father')
            g.apply_edges(self.adding, es, etype='father')
            g.send_and_recv(es, fn.copy_e('ef1', 'ef1'),
                            fn.sum('ef1', 'nf1'), etype='father')
            g.send_and_recv(es, fn.copy_e('ef2', 'ef2'),
                            fn.mean('ef2', 'nf2'), etype='father')
            g.send_and_recv(es, fn.copy_e('epwr', 'epwr'),
                            fn.sum('epwr', 'pwr'), etype='father')
            g.send_and_recv(es, fn.copy_e('epwr-feat', 'epwr-feat'),
                            fn.sum('epwr-feat', 'pwr-feat'), etype='father')
            g.apply_nodes(self.module_reduce, layer_nodes)
        return g.ndata['pwr'], g.ndata['pwr-feat']

class Itg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedx = MLP(24, 8)
        self.cnn_encoder = CNNEncoder(8)
        self.prop_test = Prop_test(8, 14, 8)

    def forward(self,x_process, label_macro, swi, itn, g, topo, cnn, masks):
        ttl_nodes = g.num_nodes()
        cell_num = swi.shape[0]
        cell_embed = self.embedx(x_process)
        cnn_res = self.cnn_encoder(cnn, masks)
        g.ndata['new_nf'] = g.ndata['nf'] = torch.cat(
            [cell_embed, cnn_res], dim=0)

        g.ndata['pwr'] = torch.zeros((ttl_nodes, 3), dtype=torch.float32).to(device=g.device)
        g.ndata['pwr'][0:cell_num, 0] = swi
        g.ndata['pwr'][0:cell_num, 1] = itn
        g.ndata['pwr'][0:cell_num, 2] = torch.zeros_like(swi)

        g.ndata['pwr-feat'] = torch.zeros((ttl_nodes, 3), dtype=torch.float32).to(device=g.device)
        g.ndata['pwr-feat'][0:cell_num, 0] = label_macro[:, 1]
        g.ndata['pwr-feat'][0:cell_num, 1] = label_macro[:, 0]
        g.ndata['pwr-feat'][0:cell_num, 2] = label_macro[:, 2]

        res, res_feat = self.prop_test(g, topo)

        return res[cell_num:,:], res_feat[cell_num:,:]
