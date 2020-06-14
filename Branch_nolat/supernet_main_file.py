import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from tensorboardX import SummaryWriter

model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}
from general_functions.crowd import Crowd
from general_functions.utils import get_logger, weights_init, load, create_directories_from_list, \
                                    check_tensor_in_list, writh_new_ARCH_to_fbnet_modeldef
import supernet_functions.lookup_table_builder as ltb
from supernet_functions.lookup_table_builder import LookUpTable
from supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
from supernet_functions.training_functions_supernet import TrainerSupernet
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


def train_supernet():
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = True

    create_directories_from_list([CONFIG_SUPERNET['logging']['path_to_tensorboard_logs']])
    logger = get_logger(CONFIG_SUPERNET['logging']['path_to_log_file'])
    writer = SummaryWriter(log_dir=CONFIG_SUPERNET['logging']['path_to_tensorboard_logs'])

    #### LookUp table consists all information about layers
    branch1_lookup_table = LookUpTable(search_space=ltb.BRANCH1_SEARCH_SPACE)
    branch2_lookup_table = LookUpTable(search_space=ltb.BRANCH2_SEARCH_SPACE)
    branch3_lookup_table = LookUpTable(search_space=ltb.BRANCH3_SEARCH_SPACE)

    #### DataLoading
    device_count = torch.cuda.device_count()
    # import pdb;pdb.set_trace()
    datasets = {x: Crowd(os.path.join(CONFIG_SUPERNET['dataloading']['dataset_path'], x),
                          CONFIG_SUPERNET['dataloading']['crop_size'],
                          CONFIG_SUPERNET['dataloading']['downsample_ratio'],
                          CONFIG_SUPERNET['dataloading']['is_gray'], x) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(datasets[x],
                          collate_fn=(train_collate if x == 'train' else default_collate),
                          batch_size=(CONFIG_SUPERNET['dataloading']['batch_size'] if x == 'train' else 1),
                          shuffle=(True if x == 'train' else False),
                          num_workers=CONFIG_SUPERNET['dataloading']['num_workers'] * device_count,
                          pin_memory=(True if x == 'train' else False))
                    for x in ['train', 'val']}

    train_w_loader = train_thetas_loader = dataloaders['train']
    test_loader = dataloaders['val']

    #### Model
    model = FBNet_Stochastic_SuperNet(branch1_lookup_table, branch2_lookup_table, branch3_lookup_table).cuda()
    model = model.apply(weights_init)
    if False:
        from collections import OrderedDict
        base_weights = torch.load("./supernet_functions/logs/ckpt_nolat.pth", map_location=torch.device('cuda'))
        new_state_dict = OrderedDict()
        for k, v in base_weights.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    # model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    model = nn.DataParallel(model, device_ids=[0,1])
    #### Loss, Optimizer and Scheduler
    criterion = SupernetLoss(CONFIG_SUPERNET['loss']['sigma'],
                             CONFIG_SUPERNET['dataloading']['crop_size'],
                             CONFIG_SUPERNET['loss']['downsample_ratio'],
                             CONFIG_SUPERNET['loss']['background_ratio'],
                             CONFIG_SUPERNET['loss']['use_background'],
                             ).cuda()

    thetas_params = [param for name, param in model.named_parameters() if 'thetas' in name]
    params_except_thetas = [param for param in model.parameters() if not check_tensor_in_list(param, thetas_params)]

    w_optimizer = torch.optim.Adam(params=params_except_thetas,
                                  lr=CONFIG_SUPERNET['optimizer']['w_lr'], 
                                  weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
    
    theta_optimizer = torch.optim.Adam(params=thetas_params,
                                       lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                       weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])


    w_scheduler = None
    #### Training Loop
    trainer = TrainerSupernet(criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer)
    trainer.train_loop(train_w_loader, train_thetas_loader, test_loader, model)


if __name__ == "__main__":
    train_supernet()
