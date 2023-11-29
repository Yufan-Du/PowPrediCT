import os
# Some Settings
test_design = "Vortex-large"
test_designs = [test_design]
data_file = "/home/yufandu/power_estim/data/route_graph_phase1/"
store_file = os.path.join("/home/yufandu/power_estim/checkpoint", test_design, "phase1/")
#model_path = ""
learning_rate = 0.0003
lr_decay_rate = 0.999

# cell driver strength
driver_type = 36

# Loop Setting
num_epochs = 10000
iternum_per_epoch = 56
iternum_per_record = 5
iter_num = 0
record = {"train": {}, "test": {}}

# Scaler and divisor are used to make the model input and output stay the right magnitude and faster convergence.
swi_p_divosor = 100000
int_p_divisor = 10000
label_multiplier = 100000

# A factor for lut table post-process, consider external conditions(temperature, voltage ...)
# For details, please check Innovus User Guide.
int_p_factor = 0.93 
