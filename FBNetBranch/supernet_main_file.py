import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from tensorboardX import SummaryWriter
from scipy.special import softmax
import argparse

from general_functions.crowd import Crowd
from general_functions.utils import get_logger, weights_init, load, create_directories_from_list, \
                                    check_tensor_in_list, writh_new_ARCH_to_fbnet_modeldef
import supernet_functions.lookup_table_builder as ltb
from supernet_functions.lookup_table_builder import LookUpTable
from supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
from supernet_functions.training_functions_supernet import TrainerSupernet
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH
    
parser = argparse.ArgumentParser("action")
parser.add_argument('--train_or_sample', type=str, default='', \
                    help='train means training of the SuperNet, sample means sample from SuperNet\'s results')
parser.add_argument('--architecture_name', type=str, default='', \
                    help='Name of an architecture to be sampled')
parser.add_argument('--hardsampling_bool_value', type=str, default='True', \
                    help='If not False or 0 -> do hardsampling, else - softmax sampling')
args = parser.parse_args()

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes, transposed_batch[4]


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
    branch1_lookup_table = LookUpTable(search_space=ltb.BRANCH1_SEARCH_SPACE,
                                       calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'],
                                       brancename="branch1.txt")
    branch2_lookup_table = LookUpTable(search_space=ltb.BRANCH2_SEARCH_SPACE,
                                       calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'],
                                       brancename="branch2.txt")
    branch3_lookup_table = LookUpTable(search_space=ltb.BRANCH3_SEARCH_SPACE,
                                       calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'],
                                       brancename="branch3.txt")

    #### DataLoading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = nn.DataParallel(model, device_ids=[0])
    #### Loss, Optimizer and Scheduler
    criterion = SupernetLoss(CONFIG_SUPERNET['loss']['sigma'],
                             CONFIG_SUPERNET['dataloading']['crop_size'],
                             CONFIG_SUPERNET['loss']['downsample_ratio'],
                             CONFIG_SUPERNET['loss']['background_ratio'],
                             CONFIG_SUPERNET['loss']['use_background'],
                             ).cuda()

    thetas_params = [param for name, param in model.named_parameters() if 'thetas' in name]
    params_except_thetas = [param for param in model.parameters() if not check_tensor_in_list(param, thetas_params)]

    w_optimizer = torch.optim.SGD(params=params_except_thetas,
                                  lr=CONFIG_SUPERNET['optimizer']['w_lr'], 
                                  momentum=CONFIG_SUPERNET['optimizer']['w_momentum'],
                                  weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
    
    theta_optimizer = torch.optim.Adam(params=thetas_params,
                                       lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                       weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])

    last_epoch = -1
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer,
                                                             T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
                                                             last_epoch=last_epoch)
    
    #### Training Loop
    trainer = TrainerSupernet(criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer)
    trainer.train_loop(train_w_loader, train_thetas_loader, test_loader, model)


if __name__ == "__main__":
    assert args.train_or_sample in ['train', 'sample']
    if args.train_or_sample == 'train':
        train_supernet()
    # elif args.train_or_sample == 'sample':
    #     assert args.architecture_name != '' and args.architecture_name not in MODEL_ARCH
    #     hardsampling = False if args.hardsampling_bool_value in ['False', '0'] else True
    #     sample_architecture_from_the_supernet(unique_name_of_arch=args.architecture_name, hardsampling=hardsampling)