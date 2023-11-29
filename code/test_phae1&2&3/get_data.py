import os
import numpy as np
import json
import dgl
import torch
import pickle
from tqdm import tqdm
from PIL import Image
from config import cnn_path

def get_data(data_folder, designs, verbose = False):
    data_pack = {}
    print("Load dataset...")
    for design in tqdm(designs):
        name = design.split("_")[0]
        freq = (int)(design.split("_")[1])
        util = (int)(design.split("_")[2])

        # label for cells
        label_module = np.load(os.path.join(data_folder, design,"label-module.npy"))
        if verbose:
            print("Loaded npy label(module):", label_module.shape)
        label_cts = np.load(os.path.join(data_folder, design,"label-clk.npy"))
        if verbose:
            print("Loaded npy label(clk):", label_cts.shape)
        label_macro = np.load(os.path.join(data_folder, design,"label.npy"))
        if verbose:
            print("Loaded npy label(macro):", label_macro.shape)
        have_swi = np.load(os.path.join(data_folder, design,"have_swi.npy"))
        if verbose:
            print("Loaded have swi:", have_swi.shape)
        masks = np.load(os.path.join(data_folder, design,"masks.npy"))
        if verbose:
            print("Loaded npy masks(module):", masks.shape)
        dri_pin = np.load(os.path.join(data_folder, design,"clk-pin.npy"))
        if verbose:
            print("Loaded dri_pin:", dri_pin.shape)
        cnn_ttl = os.path.join(cnn_path, name, "DRC/feature",f"freq_{freq}_mp_1_fpu_{util}_fpa_1.0_p_4_fi_ap.npy")
        cnn = np.load(cnn_ttl)
        cnn = np.transpose(cnn, (2,0,1))
        if verbose:
            print("Loaded npy CNN:", cnn.shape)

        # DGL
        loaded_g, _ =  dgl.load_graphs(os.path.join(data_folder, design,"dgl_data.dgl"))
        g = loaded_g[0]
        g.ndata['nf'] = torch.cat((g.ndata['netn'],g.ndata['dri'].unsqueeze(1),g.ndata['is_t'].unsqueeze(1),g.ndata['is_ck'].unsqueeze(1), \
            g.ndata['cap'], g.ndata['slew']),dim=1)

        for attr in ['netn', 'cap', 'x', 'y', 'slew', 'one-hot-type']:
            if attr in g.ndata:
                del g.ndata[attr]

        loaded_g_m, _ =  dgl.load_graphs(os.path.join(data_folder, design,"dgl_module.dgl"))
        g_m = loaded_g_m[0]

        # less data on Cuda
        g.edges['net_out'].data['nef'] = torch.cat((g.edges['net_out'].data['fourier_encoded'], g.edges['net_out'].data['hpwl_encoded'], 
                         g.edges['net_out'].data['dx_over_d'].unsqueeze(1),g.edges['net_out'].data['dy_over_d'].unsqueeze(1)), dim=1)
        g.edges['net_in'].data['nef'] = torch.cat((g.edges['net_in'].data['fourier_encoded'], g.edges['net_in'].data['hpwl_encoded'], 
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

        # TOPO sort
        with open(os.path.join(data_folder, design,"sorted_hier.json"), 'rb') as f:
            topo_raw = json.load(f)
        topo_module = [torch.tensor(data) for data in topo_raw]
        data_pack[design] = g, g_m, label_module,label_macro, topo_module, masks, cnn, have_swi, label_cts, dri_pin

    return data_pack