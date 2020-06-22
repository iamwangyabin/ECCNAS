import numpy as np

CONFIG_SUPERNET = {
    'gpu_settings' : {
        'gpu_nums' : 1,
        'use_amp' : False
    },
    'logging' : {
        'path_to_log_dir' : './supernet_functions/logs/',
        'path_to_log_file': './supernet_functions/logs/logger/',
        'path_to_tensorboard_logs': './supernet_functions/logs/tb'
    },
    'dataloading': {
        'dataset_path': "/home/yabin/FasterCrowdCountingNAS/UCF-QNRF_ECCV18",
        'UCF-QNRF_ECCV18_path': "/home/yabin/FasterCrowdCountingNAS/UCF-QNRF_ECCV18",  # 1201
        'JHU_path': "/home/teddy/JHU_Train_Val_Test",  # 4380
        'NWPU_path': "/home/teddy/NWPU_Train_Val_Test",  # 3169
        'crop_size': 512,
        'is_gray': False,
        'num_workers': 8,
        'batch_size': 1,
        'downsample_ratio': 8,
    },
    'optimizer' : {
        # SGD parameters for w
        'w_lr' : 1e-5,
        'w_weight_decay' : 1e-4,
        # Adam parameters for thetas
        'thetas_lr' : 1e-5,
        'thetas_weight_decay' : 5 * 1e-4
    },
    'loss' : {
        'alpha' : 0.2,
        'beta' : 0.1,
        'sigma': 8.0,
        'downsample_ratio': 8,
        'background_ratio': 0.15,
        'use_background': True,
    },
    'train_settings' : {
        'print_freq' : 10,
        'path_to_save_model' : './supernet_functions/logs/ckpt_nolat.pth' ,
        'cnt_epochs' : 5000,
        'train_thetas_from_the_epoch' : 1,
        # for Gumbel Softmax
        'init_temperature' : 5.0,
        'exp_anneal_rate' : np.exp(-0.045)
    }
}
