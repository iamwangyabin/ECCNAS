import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from general_functions.crowd import Crowd
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.densenet import DenseNet

def test_supernet():
    datasets = Crowd(os.path.join(CONFIG_SUPERNET['dataloading']['dataset_path'], 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = DenseNet(growthRate=32, depth=34, reduction=0.5, bottleneck=True).cuda()
    model.load_state_dict(torch.load("./supernet_functions/logs/ckpt_nolat.pth", map_location=torch.device('cuda')), strict=True)
    model = nn.DataParallel(model)
    epoch_minus = []
    for inputs, count, name in dataloader:
        inputs = inputs.to("cuda")
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs, 1)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)
    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)

if __name__ == "__main__":
    test_supernet()
