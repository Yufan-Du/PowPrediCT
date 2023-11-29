import os
import torch
import random
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from config import *
from get_data import get_data
from model import Prediction
from model_cts import Prediction_cts
from model_cnn import Itg

def post_process(g, x, x_swi_out, e, e_input, num):
    num_nodes = g.num_nodes()
    pile_swi = torch.zeros((num)).to(x.device)
    pile_swi_tog = torch.zeros((num)).to(x.device)
    x_process = torch.zeros((num, x_swi_out.shape[1])).to(x.device)
    pile_int = torch.zeros((num_nodes)).to(x.device)
    pile_int_input = torch.zeros((num_nodes)).to(x.device)
    pile_int_toggle = torch.zeros((num_nodes)).to(x.device)
    pile_int_final = torch.zeros((num)).to(x.device)
    pile_int_final_input = torch.zeros((num)).to(x.device)

    g.ndata['swi'] = x
    g.ndata['swi_out'] = x_swi_out

    src_nodes, dst_nodes = g.edges(etype='cell_out')
    e = e * g.ndata['tg'][dst_nodes] * g.ndata['tg'][src_nodes]
    e = e.to(torch.float32)
    e_input = e_input * g.ndata['tg'][src_nodes]
    e_input = e_input.to(torch.float32)
    pile_int.index_add_(0, g.edges['cell_out'].data['driver'].int(), e)
    pile_int_input.index_add_(0, g.edges['cell_out'].data['driver'].int(), e_input)
    pile_int_toggle.index_add_(0, g.edges['cell_out'].data['driver'].int(
    ), g.ndata['tg'][src_nodes].to(dtype=pile_int_toggle.dtype))
    pile_int = (pile_int + 1e-9) / (pile_int_toggle + 1e-6)

    g.ndata['int'] = pile_int
    g.ndata['int_input'] = pile_int_input

    selected_nodes = torch.where(g.ndata['dri'] > 0.5)[0]
    sub_g = g.subgraph(selected_nodes)

    pile_swi.index_add_(0, sub_g.ndata['belonging'].int(), (sub_g.ndata['swi'] * sub_g.ndata['tg']).to(dtype=pile_swi.dtype))
    pile_swi_tog.index_add_(0, sub_g.ndata['belonging'].int(), (sub_g.ndata['tg']).to(dtype=pile_swi.dtype))
    x_process.index_add_(0, sub_g.ndata['belonging'].int(), (sub_g.ndata['swi_out']).to(dtype=x_process.dtype))

    counts = torch.zeros_like(pile_swi)
    counts.index_add_(0, sub_g.ndata['belonging'].int(), torch.ones_like(sub_g.ndata['tg']))

    swi_tog_means = pile_swi_tog / (counts + 1e-6)
    pile_int_final.index_add_(
        0, sub_g.ndata['belonging'].int(), sub_g.ndata['int'])
    pile_int_final_input.index_add_(
        0, sub_g.ndata['belonging'].int(), sub_g.ndata['int_input'])
    return pile_swi, swi_tog_means, x_process, pile_int_final + pile_int_final_input

# loss function
def compute_loss_stage2(pwr, pwr_feat, labels, topo_module, design_name, cts_tensor, label_cts):
    cell_num = len(topo_module[0])
    weight = torch.zeros((pwr.shape[0], 1), dtype=torch.float32).to(pwr.device)
    for i in range(1, len(topo_module)):
        weight[topo_module[i]-cell_num] = 1/ (i**6)
    swi_p, int_p, lea_p = torch.split(pwr,[1,1,1],dim=1)
    swi_p = (swi_p / scale_estim).squeeze(1) * swi_res_portion  + pwr_feat[:,0] * early_feature_multiplier * swi_early_portion
    int_p = (int_p / scale_estim_int).squeeze(1) * int_res_portion + pwr_feat[:,1] * early_feature_multiplier * int_early_portion
    lea_p =  (lea_p / scale_lea).squeeze(1) * lea_res_portion + pwr_feat[:,2] * early_feature_multiplier * lea_early_portion
    cts_tensor = (cts_tensor / scale_estim_cts)

    label_int = labels[:,0:1].sum(dim=1).to(pwr.device) * label_multiplier # int
    label_swi = labels[:,1:2].sum(dim=1).to(pwr.device) * label_multiplier # swi
    label_lea = labels[:,2:3].sum(dim=1).to(pwr.device) * label_multiplier # lea
    label_swi_cts = label_cts[:,1:2].sum(dim=1).to(pwr.device) * label_multiplier # cts

    loss_int = torch.nn.functional.l1_loss(int_p, label_int, reduction='mean')
    loss_swi = torch.nn.functional.l1_loss(swi_p*weight, label_swi*weight, reduction='mean')
    loss_swi_cts = torch.nn.functional.l1_loss(cts_tensor, label_swi_cts, reduction='mean')
    loss_lea = torch.nn.functional.l1_loss(lea_p*weight, label_lea*weight, reduction='mean')

    sum_estim_int = int_p[0]
    sum_estim_swi = swi_p[0]
    sum_estim_swi_cts = cts_tensor.sum(dim=0)
    sum_estim_lea = lea_p[0]
    sum_label_int = label_int[0]
    sum_label_swi = label_swi[0]
    sum_label_swi_cts = label_swi_cts.sum(dim=0)
    sum_label_lea = label_lea[0]
    return loss_int, loss_swi,loss_lea, loss_swi_cts ,sum_estim_int, sum_label_int,sum_estim_swi, sum_label_swi,sum_estim_lea,sum_label_lea,sum_estim_swi_cts,sum_label_swi_cts

folders = os.listdir(data_file)
folders = [folder for folder in folders if folder.split('_')[1]]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Use device: ", device)

model = Prediction(ndim_driver=driver_type).to(device)
model_itg = Itg().to(device)
model_cts = Prediction_cts(ndim_driver=driver_type).to(device)

map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(os.path.join(model_path, model_choice1), map_location=map_location))

for param in model.parameters():
    param.requires_grad = False
optimizer = optim.Adam(list(model_itg.parameters()) + list(model_cts.parameters()), lr=learning_rate)
optimizer_stage1 = optim.Adam(model.parameters(), lr=learning_rate/10)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=lr_decay_rate)


swi_err = []
int_err = []
lea_err = []
ttl_err = []
swi_err_test = []
int_err_test = []
lea_err_test = []
ttl_err_test = []

epoch_choice = random.sample(folders, iternum_per_epoch)
# For test
# epoch_choice = ["zero-riscy_200_50_place"]
data_pack = get_data(data_file, epoch_choice ,verbose=False)
for epoch in range(num_epochs):
    random.shuffle(epoch_choice)
    for f in epoch_choice:
        #print("Current file:", f)
        design_name = f.split("_")[0]
        design_freq = f.split("_")[1]
        design_util = f.split("_")[2]
        record_name = design_name+"_"+design_freq
        g, g_m, label_module,label_macro, topo_module, masks, cnn, have_swi, label_cts, dri_pin = data_pack[f]
        g = g.to(device)
        g_m = g_m.to(device)
        topo_module = [layer.to(g.device) for layer in topo_module]
        masks = torch.from_numpy(masks).to(device).to(dtype=torch.float32)
        cnn = torch.from_numpy(cnn).to(device)
        labels = torch.from_numpy(label_module).to(device)
        label_macro = torch.from_numpy(label_macro).to(device)
        have_swi = torch.from_numpy(have_swi).to(device)
        label_cts = torch.from_numpy(label_cts).to(device)
        dri_pin = torch.from_numpy(dri_pin).to(device)

        if any(f.split("_")[0]==prefix for prefix in test_designs):
            start_time = time.time()
            model.eval()
            model_itg.eval()
            model_cts.eval()
            with torch.no_grad():
                # Stage one
                x, x_swi_out,e, e_input = model(g,design_freq, design_util)
                x_cts, x_swi_out_cts = model_cts(g,design_freq, design_util, dri_pin)

                x[dri_pin] = x_cts[dri_pin]
                x_swi_out[dri_pin] = x_swi_out_cts[dri_pin]
                swi_p, swi_tog, x_process, int_p = post_process(g, x, x_swi_out, e, e_input, label_macro.shape[0])

                x_tensor = torch.zeros_like(swi_p)
                x_tensor[have_swi == 0] = 0
                x_tensor[have_swi == 1] = swi_p[have_swi == 1]
                cts_tensor = swi_p[have_swi == 0]
                # Stage two
                pwr, pwr_feat = model_itg(x_process, label_macro, x_tensor, int_p, g_m, topo_module, cnn, masks)   
                loss_int, loss_swi, loss_lea,loss_cts, sum_data, sum_label, sum_data2, sum_label2 ,sum_data3, sum_label3, sum_data4, sum_label4 = compute_loss_stage2(pwr, pwr_feat,labels, topo_module, f, cts_tensor, label_cts)

                print("Pred:", sum_data.item()," ",sum_data2.item()," ", sum_data3.item()," ", sum_data4.item())
                print("Label:", sum_label.item()," ",sum_label2.item()," ", sum_label3.item()," ", sum_label4.item())
                ttl_error = torch.abs(sum_data - sum_label)/sum_label*100
                ttl_error2 = torch.abs(sum_data2 - sum_label2)/sum_label2*100
                ttl_error3 = torch.abs(sum_data3 - sum_label3)/sum_label3*100
                ttl_error4 = torch.abs(sum_data4 - sum_label4)/sum_label4*100
                ttl_error_swi = torch.abs(sum_data2 + sum_data4 - sum_label2 - sum_label4) /(sum_label2 + sum_label4)*100
                ttl = torch.abs(sum_data+sum_data2+sum_data3+sum_data4-sum_label-sum_label2-sum_label3-sum_label4)/(sum_label+sum_label2+sum_label3+sum_label4)*100
                int_err_test.append(ttl_error.detach().cpu().numpy())
                swi_err_test.append(ttl_error2.detach().cpu().numpy())
                lea_err_test.append(ttl_error3.detach().cpu().numpy())
                ttl_err_test.append(ttl.detach().cpu().numpy())
                print(f'Epoch {epoch+1}/{num_epochs}({f}). Err(Int):{ttl_error}%,Err(Swi):{ttl_error2.item()}%,Err(Lea):{ttl_error3.item()}%,Err(with CTS):{ttl_error_swi.item()}%,Err(T):{ttl.item()}%.Test.')
            if record_name not in record['test']:
                record['test'][record_name] = {"Internal": [], "Switching": [],"Leakage": [],"Total":[]}
            record['test'][record_name]["Internal"].append(ttl_error.detach().cpu().numpy())
            record['test'][record_name]["Switching"].append(ttl_error2.detach().cpu().numpy())
            record['test'][record_name]["Leakage"].append(ttl_error3.detach().cpu().numpy())
            record['test'][record_name]["Total"].append(ttl.detach().cpu().numpy())

            end_time = time.time()
            print(f"Infer time: {end_time - start_time}")
        else:
            model.train()
            model_itg.train()
            model_cts.train()
            optimizer.zero_grad()
            optimizer_stage1.zero_grad()

            x,  x_swi_out,e, e_input = model(g,design_freq, design_util)
            x_cts, x_swi_out_cts = model_cts(g,design_freq, design_util, dri_pin)
            x[dri_pin] = x_cts[dri_pin]
            x_swi_out[dri_pin] = x_swi_out_cts[dri_pin]
            swi_p, swi_tog, x_process ,int_p = post_process(g, x, x_swi_out, e, e_input, label_macro.shape[0])

            x_tensor = torch.ones_like(swi_p)
            x_tensor[have_swi == 0] = 0
            x_tensor[have_swi == 1] = swi_p[have_swi == 1]
            cts_tensor = swi_p[have_swi == 0]

            # Stage two
            pwr, pwr_feat = model_itg(x_process ,label_macro, x_tensor, int_p, g_m, topo_module, cnn, masks)

            loss_int, loss_swi, loss_lea, loss_cts, sum_data, sum_label, sum_data2, sum_label2 ,sum_data3, sum_label3, sum_data4, sum_label4 = compute_loss_stage2(pwr, pwr_feat, labels, topo_module, f, cts_tensor, label_cts)
            ttl_error = torch.abs(sum_data - sum_label)/sum_label*100
            ttl_error2 = torch.abs(sum_data2 - sum_label2)/sum_label2*100
            ttl_error3 = torch.abs(sum_data3 - sum_label3)/sum_label3*100
            ttl_error4 = torch.abs(sum_data4 - sum_label4)/sum_label4*100
            ttl_error_swi = torch.abs(sum_data2 + sum_data4 - sum_label2 - sum_label4) /(sum_label2 + sum_label4)*100
            ttl = torch.abs(sum_data+sum_data2+sum_data3+sum_data4-sum_label-sum_label2-sum_label3-sum_label4)/(sum_label+sum_label2+sum_label3+sum_label4)*100
            int_err.append(ttl_error.detach().cpu().numpy())
            swi_err.append(ttl_error2.detach().cpu().numpy())
            lea_err.append(ttl_error3.detach().cpu().numpy())
            ttl_err.append(ttl.detach().cpu().numpy())
            total_loss = ((200-epoch)/50  + 4)* (loss_int + loss_swi +  loss_cts) + 4 * ttl_error3# + ttl_error  + ttl_error2 + ttl_error3 + ttl_error4
            total_loss.backward()
            optimizer.step()
            optimizer_stage1.step()

            print("Pred:", sum_data.item()," ",sum_data2.item()," ", sum_data3.item()," ", sum_data4.item())
            print("Label:", sum_label.item()," ",sum_label2.item()," ", sum_label3.item()," ", sum_label4.item())
            print(f'Epoch {epoch+1}/{num_epochs}({f}). Err(Int):{ttl_error}%,Err(Swi):{ttl_error2.item()}%,Err(Lea):{ttl_error3.item()}%,Err(with CTS):{ttl_error_swi.item()}%,Err(T):{ttl.item()}%.Train.')
            if record_name not in record['train']:
                record['train'][record_name] = {"Internal": [], "Switching": [], "Leakage": [], "Total": []}
            record['train'][record_name]["Internal"].append(ttl_error.detach().cpu().numpy())
            record['train'][record_name]["Switching"].append(ttl_error2.detach().cpu().numpy())
            record['train'][record_name]["Leakage"].append(ttl_error3.detach().cpu().numpy())
            record['train'][record_name]["Total"].append(ttl.detach().cpu().numpy())
        torch.cuda.empty_cache()
        iter_num += 1
    scheduler.step()
    if epoch>0 and (epoch + 1) % iternum_per_record== 0:
        fig, ax = plt.subplots()
        int_err_np = [item for item in int_err]
        swi_err_np = [item for item in swi_err]
        lea_err_np = [item for item in lea_err]
        ttl_err_np = [item for item in ttl_err]

        int_err_test_np = [item for item in int_err_test]
        swi_err_test_np = [item for item in swi_err_test]
        lea_err_test_np = [item for item in lea_err_test]
        ttl_err_test_np = [item for item in ttl_err_test]
        ax.boxplot([int_err_np, swi_err_np,lea_err_np,ttl_err_np, int_err_test_np, swi_err_test_np,lea_err_test_np,ttl_err_test_np], vert=False)
        ax.set_yticklabels(['Int. Trn.', 'Swi. Trn.','Lkg. Trn.','Ttl. Trn.','Int. Tst.', 'Swi. Tst.','Lkg. Tst.','Ttl. Tst.'])
        ax.set_xlabel('(%)')
        int_err= []
        swi_err= []
        lea_err= []
        ttl_err= []
        int_err_test= []
        swi_err_test= []
        lea_err_test= []
        ttl_err_test= []
        record = {"train":{}, "test":{}}
        plt.savefig(f'{store_file}epoch{epoch+1}.png')
        #torch.save(model.state_dict(), f"{store_file}model_epoch_{epoch}.pth")
        torch.save(model_itg.state_dict(), f"{store_file}model_cnn_epoch_{epoch+1}.pth")
        torch.save(model_cts.state_dict(), f"{store_file}model_cts_epoch_{epoch+1}.pth")