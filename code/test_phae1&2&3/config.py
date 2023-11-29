# Some Settings
test_list = ["Vortex-large", "Vortex-small", "openc910-1", "nvdla-small", "RISCY", "RISCY-FPU", "zero-riscy"]
cnn_path = "/home/yufandu/power_estim/data/cnn_phase3/"
data_file = "/home/yufandu/power_estim/data/place_graph_phase2&3/"

model_path1_1 = "/home/yufandu/power_estim/checkpoint"
model_path1_2 = "phase1"
model_choice1 = "model_epoch_150.pth"
model_path2_1 = "/home/yufandu/power_estim/checkpoint"
model_path2_2 = "phase2&3"
model_choice2 = "model_cnn_epoch_2.pth"
model_choice3 = "model_cts_epoch_2.pth"

# cell driver strength
driver_type = 36

# Loop Setting
iter_num = 0
record = {"test": {}}

# Scaler and divisor are used to make the model input and output stay the right magnitude and faster convergence.
cts_factor = 1.2
int_factor = 1.03
swi_p_divosor = 1000000
int_p_divisor = 10000
label_multiplier = 100000
scale_estim = 1000000
scale_estim_int = 10000 * int_factor
scale_estim_cts = 10000 / cts_factor
early_feature_multiplier = 100000
scale_lea = 50

# The final power could be the combination of early analysis report and residual part.
# This parameters can be set as learnable parameters. 
# For simplicity, we've picken several good starting point for training to faster the training process on CircuitNet.
# It may not work for other dataset, so use them carefully...
swi_early_portion = 0.77
int_early_portion = 0.7
lea_early_portion = 1.05
swi_res_portion = 0.40
int_res_portion = 0.3
lea_res_portion = 1

# A factor for lut table post-process, consider external conditions(temperature, voltage ...)
# For details, please check Innovus User Guide.
int_p_factor = 0.93 
