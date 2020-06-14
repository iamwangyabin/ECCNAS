import numpy as np

CONFIG_SUPERNET = {
    'gpu_settings' : {
        'gpu_nums' : 1,
        'use_amp' : False
    },
    'lookup_table' : {
        'create_from_scratch' : False,
        'path_to_lookup_table' : './supernet_functions/lookup_table_',
        'number_of_runs' : 5 # each operation run number_of_runs times and then we will take average
    },
    'logging' : {
        'path_to_tensorboard_logs' : './supernet_functions/logs/',
        'path_to_log_dir' : './supernet_functions/logs/',
        'path_to_log_file' : 'log'
    },
    'dataloading': {
        'dataset_path': r"D:\Bayesian-Crowd-Counting-master\processedData",
        'crop_size': 256,
        'is_gray': False,
        'num_workers': 1,
        'batch_size': 1,
        'downsample_ratio': 8,
    },
    'optimizer' : {
        # SGD parameters for w
        'w_lr' : 0.01,
        'w_momentum' : 0.9,
        'w_weight_decay' : 1e-4,
        # Adam parameters for thetas
        'thetas_lr' : 0.01,
        'thetas_weight_decay' : 5 * 1e-4
    },
    'loss' : {
        'alpha' : 0.2,
        'beta' : 0.6,
        'sigma': 8.0,
        'downsample_ratio': 8,
        'background_ratio': 0.15,
        'use_background': True,
    },
    'train_settings' : {
        'print_freq' : 50,
        'path_to_save_model' : './ckpt',
        'cnt_epochs' : 500,
        'train_thetas_from_the_epoch' : 10,
        # for Gumbel Softmax
        'init_temperature' : 5.0,
        'exp_anneal_rate' : np.exp(-0.045)
    }
}