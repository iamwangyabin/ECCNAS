import torch
import supernet_functions.lookup_table_builder as ltb
from supernet_functions.lookup_table_builder import LookUpTable
from supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
branch1_lookup_table = LookUpTable(search_space=ltb.BRANCH1_SEARCH_SPACE,
                                   calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'],
                                   brancename="branch1.txt")
branch2_lookup_table = LookUpTable(search_space=ltb.BRANCH2_SEARCH_SPACE,
                                   calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'],
                                   brancename="branch2.txt")
branch3_lookup_table = LookUpTable(search_space=ltb.BRANCH3_SEARCH_SPACE,
                                   calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'],
                                   brancename="branch3.txt")
model = FBNet_Stochastic_SuperNet(branch1_lookup_table, branch2_lookup_table, branch3_lookup_table)


from collections import OrderedDict   #导入此模块

base_weights = torch.load(r'C:\Users\38623\Documents\GitHub\FasterCrowdCountingNAS\FBNetBranch\ckpt.pth', map_location=torch.device('cuda'))
new_state_dict = OrderedDict()
for k, v in base_weights.items():
    name = k[7:]  # remove `vgg.`，即只取vgg.0.weights的后面几位
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)


import numpy as np
ops_names = [op_name for op_name in branch1_lookup_table.lookup_table_operations]
arch_operations=[]
for layer in model.branch3_to_search:
    arch_operations.append(ops_names[np.argmax(layer.thetas.detach().cpu().numpy())])

arch_operations

arch_operations=[]
for layer in model.branch3_to_search:
    arch_operations.append(layer.thetas.detach().cpu().numpy())

arch_operations