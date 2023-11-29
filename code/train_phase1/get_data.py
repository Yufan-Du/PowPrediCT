import os
import numpy as np
import dgl
import torch
from tqdm import tqdm

def get_data(data_folder, designs, verbose = False):
    data_pack = {}
    print("Load dataset...")
    for design in tqdm(designs):
        # labels for cell-wise training
        label = np.load(os.path.join(data_folder, design,"label.npy"))
        if verbose:
            print("Loaded npy label:", label.shape)

        # DGL
        loaded_g, _ =  dgl.load_graphs(os.path.join(data_folder, design,"dgl_data.dgl"))
        g = loaded_g[0]
        g.ndata['nf'] = torch.cat((g.ndata['netn'],g.ndata['dri'].unsqueeze(1),g.ndata['is_t'].unsqueeze(1),g.ndata['is_ck'].unsqueeze(1), \
            g.ndata['cap'], g.ndata['slew']),dim=1)

        # less data on Cuda 
        for attr in ['netn', 'cap', 'x', 'y', 'slew', 'one-hot-type']:
            if attr in g.ndata:
                del g.ndata[attr]

        g.edges['net_out'].data['nef'] = torch.cat((g.edges['net_out'].data['fourier_encoded'], g.edges['net_out'].data['hpwl'].unsqueeze(1), 
                         g.edges['net_out'].data['dx_over_d'].unsqueeze(1),g.edges['net_out'].data['dy_over_d'].unsqueeze(1)), dim=1)
        g.edges['net_in'].data['nef'] = torch.cat((g.edges['net_in'].data['fourier_encoded'], g.edges['net_in'].data['hpwl'].unsqueeze(1), 
                         g.edges['net_in'].data['dx_over_d'].unsqueeze(1),g.edges['net_in'].data['dy_over_d'].unsqueeze(1)), dim=1)

        g.edges['cell_out'].data['ef'] = torch.cat((g.edges['cell_out'].data['lut'].unsqueeze(1), g.edges['cell_out'].data['index1'], g.edges['cell_out'].data['index2'], \
            g.edges['cell_out'].data['values'].to(torch.float32)), dim=1)
        g.edges['cell_out'].data['ef_input'] = torch.cat((g.edges['cell_out'].data['lut_input'].unsqueeze(1), g.edges['cell_out'].data['index1_input'], \
            g.edges['cell_out'].data['values_input'].to(torch.float32)), dim=1)

        # less data on Cuda
        attributes_to_remove = ['fourier_encoded', 'hpwl_encoded', 'dx_over_d', 'dy_over_d']
        for attr in attributes_to_remove:
            if attr in g.edges['net_out'].data:
                del g.edges['net_out'].data[attr]
 
        for attr in attributes_to_remove:
            if attr in g.edges['net_in'].data:
                del g.edges['net_in'].data[attr]

        attributes_to_remove = ['lut', 'index1', 'index2', 'values', 'lut_input', 'index1_input', 'values_input']
        for attr in attributes_to_remove:
            if attr in g.edges['cell_out'].data:
                del g.edges['cell_out'].data[attr]

        data_pack[design] = g, label
    return data_pack