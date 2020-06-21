import timeit
import torch
from collections import OrderedDict
import gc
from fbnet_building_blocks.fbnet_builder import PRIMITIVES, PRIMITIVES_norm
from general_functions.utils import add_text_to_file, clear_files_in_the_list
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

# the settings from the page 4 of https://arxiv.org/pdf/1812.03443.pdf
#### table 2
CANDIDATE_BLOCKS = ["ir_k3_e1", "ir_k3_s2", "ir_k3_e3",
                    "ir_k3_e6", "ir_k5_e1", "ir_k5_s2",
                    "ir_k5_e3", "ir_k5_e6", "skip"]

CANDIDATE_BLOCKS2 = ["k3_se", "k5_se", "k7_se",
                    "k3_", "k5_", "k7_", "skip"]

# Low level
BRANCH1_SEARCH_SPACE = OrderedDict([
    ("input_shape", [(16, 512, 512),
                     (16, 512, 512),
                     (32, 512, 512),
                     (32, 512, 512)]),
    ("channel_size", [16,
                      32,
                      32,
                      32]),
    ("strides", [1,
                 1,
                 1,
                 1])
])

BRANCH2_SEARCH_SPACE = OrderedDict([
    ("input_shape", [(16, 512, 512),
                     (16, 512, 512), (32, 256, 256),
                     (32, 128, 128)]),
    ("channel_size", [16,
                      32,  32,
                      64]),
    ("strides", [1,
                 2, 2,
                 2])
])

BRANCH3_SEARCH_SPACE = OrderedDict([
    ("input_shape", [(16, 512, 512),
                     (16, 512, 512), (32, 256, 256), (32, 256, 256), (32, 256, 256),
                     (32, 256, 256), (64, 128, 128), (64, 128, 128), (64, 128, 128),
                     (64, 128, 128), (128, 64, 64), (128, 64, 64), (128, 64, 64),
                     (128, 64, 64),  (256, 64, 64), (256, 64, 64), (256, 64, 64),
                     (256, 64, 64),  (512, 32, 32), (512, 32, 32), (512, 32, 32),
                     (512, 32, 32)]),
    # table 1. filter numbers over the 22 layers
    ("channel_size", [16,
                      32, 32, 32, 32,
                      64, 64, 64, 64,
                      128, 128, 128, 128,
                      256, 256, 256, 256,
                      512, 512, 512, 512,
                      512]),
    # table 1. strides over the 22 layers
    ("strides", [1,
                 2, 1, 1, 1,
                 2, 1, 1, 1,
                 2, 1, 1, 1,
                 1, 1, 1, 1,
                 2, 1, 1, 1,
                 1])
])


class LookUpTable:
    def __init__(self, search_space, candidate_blocks=CANDIDATE_BLOCKS2):
        self.cnt_layers = len(search_space["input_shape"])
        self.lookup_table_operations = {op_name : PRIMITIVES_norm[op_name] for op_name in candidate_blocks}
        self.layers_parameters, self.layers_input_shapes = self._generate_layers_parameters(search_space)

    def _generate_layers_parameters(self, search_space):
        # layers_parameters are : C_in, C_out, expansion, stride
        layers_parameters = [(search_space["input_shape"][layer_id][0],
                              search_space["channel_size"][layer_id],
                              -999,
                              search_space["strides"][layer_id]
                             ) for layer_id in range(self.cnt_layers)]
        
        # layers_input_shapes are (C_in, input_w, input_h)
        layers_input_shapes = search_space["input_shape"]
        
        return layers_parameters, layers_input_shapes