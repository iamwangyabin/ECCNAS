CONFIG_ARCH = {
    'logging' : {
        'path_to_log_file' : './architecture_functions/logs/logger/',
        'path_to_tensorboard_logs' : './architecture_functions/logs/tb'
    },
    'dataloading' : {
        'dataset_path' : r".\UCF-QNRF_ECCV18",
        'crop_size' : 512,
        'is_gray' : False,
        'num_workers' : 8,
        'batch_size' : 3,
    },
    'optimizer' : {
        'lr' : 0.1,
        'momentum' : 0.9,
        'weight_decay' : 4 * 1e-5
    },
    'train_settings' : {
        'print_freq' : 50, # show logging information
        'path_to_save_model' : './architecture_functions/logs/best_model.pth',
        'cnt_epochs' : 140, #
        # YOU COULD USE 'CosineAnnealingLR' or 'MultiStepLR' scheduler
        'scheduler' : 'MultiStepLR',
        ## CosineAnnealingLR settings
        'eta_min' : 0.001,
        ## MultiStepLR settings
        'milestones' : [90, 110], # [90, 180, 270], # decay 10x at 90, 180, and 270 epochs
        'lr_decay' : 0.1
    },
    'loss': {
        'sigma': 8.0,
        'downsample_ratio': 8,
        'background_ratio': 0.15,  #
        'use_background': True,
    }
}