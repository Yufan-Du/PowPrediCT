import os
import torch
import random
import torch.optim as optim
from config import *
from get_data import get_data
from model import Prediction
import matplotlib.pyplot as plt
import pandas as pd

def post_process(g, x, e, e_input, num):
    num_nodes = g.num_nodes()
    pile_swi = torch.zeros((num)).to(x.device)
    pile_swi_tog = torch.zeros((num)).to(x.device)
    pile_int = torch.zeros((num_nodes)).to(x.device)
    pile_int_input = torch.zeros((num_nodes)).to(x.device)
    pile_int_toggle = torch.zeros((num_nodes)).to(x.device)
    pile_int_final = torch.zeros((num)).to(x.device)
    pile_int_final_input = torch.zeros((num)).to(x.device)
    g.ndata['swi'] = x

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
    pile_swi.index_add_(0, sub_g.ndata['belonging'].int(
    ), (sub_g.ndata['swi'] * sub_g.ndata['tg']).to(dtype=pile_swi.dtype))

    pile_swi_tog.index_add_(0, sub_g.ndata['belonging'].int(), (sub_g.ndata['tg']).to(dtype=pile_swi.dtype))

    counts = torch.zeros_like(pile_swi)
    counts.index_add_(0, sub_g.ndata['belonging'].int(), torch.ones_like(sub_g.ndata['tg']))

    swi_tog_means = pile_swi_tog / (counts + 1e-6)
    pile_int_final.index_add_(
        0, sub_g.ndata['belonging'].int(), sub_g.ndata['int'])
    pile_int_final_input.index_add_(
        0, sub_g.ndata['belonging'].int(), sub_g.ndata['int_input'])

    return pile_swi, swi_tog_means, pile_int_final + pile_int_final_input

# loss function
def compute_loss(swi_p, swi_tog, int_p, labels):
    swi_p /= swi_p_divosor
    int_p /= int_p_divisor
    int_p *= int_p_factor
    label_int = labels[:, 0:1].sum(dim=1).to(int_p.device) * label_multiplier  # int
    label_swi = labels[:, 1:2].sum(dim=1).to(swi_p.device) * label_multiplier  # swi
    int_p = int_p * labels[:, 3]
    loss_int = torch.nn.functional.l1_loss(int_p, label_int, reduction='mean')
    loss_swi = torch.nn.functional.l1_loss(swi_p_divosor * swi_p * torch.log(label_swi + 1)/(1+swi_tog), \
                                            swi_p_divosor * label_swi* torch.log(label_swi + 1)/(swi_tog+1), reduction='mean')

    sum_estim_int = int_p.sum(dim=0)
    sum_estim_swi = swi_p.sum(dim=0)
    sum_label_int = label_int.sum(dim=0)
    sum_label_swi = label_swi.sum(dim=0)
    return loss_int, loss_swi, sum_estim_int, sum_label_int, sum_estim_swi, sum_label_swi

# Get design list
folders = os.listdir(data_file)
folders = [folder for folder in folders if folder.split('_')[1]]

# Start training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Use device: ", device)

model = Prediction(ndim_driver = driver_type).to(device)
#model.load_state_dict(torch.load(f"{model_path}/{model_choice1}"))
#model_res.load_state_dict(torch.load(f"{model_path}/{model_choice2}"))

# Set learning rate and lr decay rate.
optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)

swi_err = []
int_err = []
ttl_err = []
swi_err_test = []
int_err_test = []
ttl_err_test = []

epoch_choice = random.sample(folders, iternum_per_epoch)

# For debug
#epoch_choice = ["zero-riscy_200_55_route"]
#epoch_choice = ["Vortex-small_200_55_route","Vortex-large_200_55_route"]

data_pack = get_data(data_file, epoch_choice, verbose=False)
for epoch in range(num_epochs):
    for f in epoch_choice:
        # print("Current file:", f)
        design_name = f.split("_")[0]
        design_freq = f.split("_")[1]
        design_util = f.split("_")[2]
        record_name = design_name+"_"+design_freq
        g, labels = data_pack[f]
        g = g.to(device)
        labels = torch.from_numpy(labels).to(device)

        if any(f.split("_")[0]==prefix for prefix in test_designs):
            model.eval()
            with torch.no_grad():
                x, e, e_input = model(g,design_freq, design_util)
                swi_p, swi_tog, int_p = post_process(g, x, e, e_input, labels.shape[0])
                loss_int, loss_swi, sum_data, sum_label, sum_data2, sum_label2 = compute_loss(
                    swi_p, swi_tog, int_p, labels)

                label_swi = labels[:, 1:2].sum(dim=1) * label_multiplier
                label_int_p = labels[:, 0:1].sum(dim=1) * label_multiplier

                data = {
                    "swi_p": swi_p.cpu().numpy(),
                    "int_p": int_p.cpu().numpy(),
                    "label_swi": label_swi.cpu().numpy(),
                    "label_int_p": label_int_p.cpu().numpy()
                }
                df = pd.DataFrame(data)

                df.to_csv(f"/home/yufandu/power_estim/test-large/{epoch + 1}_{f}.csv", index=False)   

                ttl_error = torch.abs(sum_data - sum_label)/sum_label*100
                ttl_error2 = torch.abs(sum_data2 - sum_label2)/sum_label2*100
                ttl = torch.abs(sum_data+sum_data2-sum_label -
                                sum_label2)/(sum_label+sum_label2)*100
                int_err_test.append(ttl_error.detach().cpu().numpy())
                swi_err_test.append(ttl_error2.detach().cpu().numpy())
                ttl_err_test.append(ttl.detach().cpu().numpy())
                print(
                    f'Epoch {epoch+1}/{num_epochs}({f}). Err(Int):{ttl_error}% .Err(Swi):{ttl_error2.item()}% .Err(T):{ttl.item()}%.Test.')
            if record_name not in record['test']:
                record['test'][record_name] = {
                    "Internal": [], "Switching": [], "Total": []}
            record['test'][record_name]["Internal"].append(
                ttl_error.detach().cpu().numpy())
            record['test'][record_name]["Switching"].append(
                ttl_error2.detach().cpu().numpy())
            record['test'][record_name]["Total"].append(
                ttl.detach().cpu().numpy())
        else:
            model.train()
            optimizer.zero_grad()
            # print(g.ndata['nf'][:40][:])
            x, e, e_input = model(g,design_freq, design_util)
            swi_p,swi_tog,  int_p = post_process(g, x, e, e_input, labels.shape[0])
            loss_int, loss_swi, sum_data, sum_label, sum_data2, sum_label2 = compute_loss(
                swi_p, swi_tog, int_p, labels)
            ttl_error = torch.abs(sum_data - sum_label)/sum_label*100
            ttl_error2 = torch.abs(sum_data2 - sum_label2)/sum_label2*100
            ttl = torch.abs(sum_data+sum_data2-sum_label -
                            sum_label2)/(sum_label+sum_label2)*100
            int_err.append(ttl_error.detach().cpu().numpy())
            swi_err.append(ttl_error2.detach().cpu().numpy())
            ttl_err.append(ttl.detach().cpu().numpy())
            total_loss = (loss_int + loss_swi) * \
                ((200-epoch)/50 + 4) + ttl_error2 * 3
            total_loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}({f}). Err(Int):{ttl_error}%. Err(Swi):{ttl_error2.item()}%. Err(T):{ttl.item()}%.Train.')
            iter_num += 1
            if record_name not in record['train']:
                record['train'][record_name] = {
                    "Internal": [], "Switching": [], "Leakage": [], "Total": []}
            record['train'][record_name]["Internal"].append(
                ttl_error.detach().cpu().numpy())
            record['train'][record_name]["Switching"].append(
                ttl_error2.detach().cpu().numpy())
            record['train'][record_name]["Total"].append(
                ttl.detach().cpu().numpy())
        torch.cuda.empty_cache()
    scheduler.step()
    if epoch > 0 and (epoch + 1) % iternum_per_record == 0:
        # you need to write down something here
        fig, ax = plt.subplots()
        int_err_np = [item for item in int_err]
        swi_err_np = [item for item in swi_err]
        ttl_err_np = [item for item in ttl_err]

        int_err_test_np = [item for item in int_err_test]
        swi_err_test_np = [item for item in swi_err_test]
        ttl_err_test_np = [item for item in ttl_err_test]
        ax.boxplot([int_err_np, swi_err_np, ttl_err_np, int_err_test_np,
                   swi_err_test_np, ttl_err_test_np], vert=False)
        ax.set_yticklabels(['Int. Trn.', 'Swi. Trn.', 'Ttl. Trn.',
                           'Int. Tst.', 'Swi. Tst.', 'Ttl. Tst.'])
        ax.set_xlabel('(%)')
        int_err = []
        swi_err = []
        ttl_err = []
        int_err_test = []
        swi_err_test = []
        ttl_err_test = []

        avg_record = {"train": {}, "test": {}}

        for phase in ['train', 'test']:
            for design_name, errors in record[phase].items():
                avg_internal = sum(errors['Internal']) / \
                    len(errors['Internal'])
                avg_switching = sum(
                    errors['Switching']) / len(errors['Switching'])
                avg_ttl = sum(errors['Total']) / len(errors['Total'])

                avg_record[phase][design_name] = {
                    'Internal': avg_internal,
                    'Switching': avg_switching,
                    'Total': avg_ttl
                }
        record = {"train": {}, "test": {}}
        plt.savefig(f'{store_file}epoch{epoch + 1}.png')
        torch.save(model.state_dict(), f"{store_file}model_epoch_{epoch + 1}.pth")