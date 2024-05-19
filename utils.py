from math import ceil
import torch.nn as nn
import re
from custom_op.conv_avg import Conv2dAvg
from custom_op.conv_avg_batch import Conv2d_Avg_Batch
from custom_op.conv_svd_with_var import Conv2dSVD_with_var
from custom_op.conv_hosvd_with_var import Conv2dHOSVD_with_var
import torch

############################### My functions ###############################################

class Hook: # Lưu lại các input/output size của mô hình
    def __init__(self, module, special_param):
        self.module = module
        self.input_size = torch.zeros(4)
        self.output_size = torch.zeros(4)
        self.special_param = special_param
        self.inputs = []#torch.empty(0, 4)
        self.outputs= []
        
        self.active = True
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        if not self.active:
            return
        # self.input_size = input[0].shape
        # self.output_size = output.shape

        self.inputs.append(input[0])
        self.outputs.append(output)
    def activate(self, active):
        self.active = active
    def remove(self):
        self.active = False
        self.hook.remove()
        # print("Hook is removed")

def attach_hooks_for_conv(module, name, hook, special_param=None):
    hook[name] = Hook(module, special_param)