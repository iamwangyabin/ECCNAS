import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
from pytorch_lightning import Trainer
from supernet_functions.searcher import NAS
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

if __name__ == '__main__':
    system = NAS(CONFIG_SUPERNET)

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'{CONFIG_SUPERNET["logging"]["path_to_log_dir"]}/ckpts',
                                                                '{epoch:02d}'),
                                          monitor='val_mse',
                                          mode='min',
                                          save_top_k=5, )

    logger = TestTubeLogger(
        save_dir=CONFIG_SUPERNET['logging']['path_to_log_dir'],
        name=CONFIG_SUPERNET['logging']['path_to_log_file'],
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=CONFIG_SUPERNET['gpu_settings']['gpu_nums'],
                      distributed_backend='ddp' if CONFIG_SUPERNET['gpu_settings']['gpu_nums'] > 1 else None,
                      num_sanity_val_steps=0 if CONFIG_SUPERNET['gpu_settings']['gpu_nums'] > 1 else 5,
                      benchmark=True,
                      precision=16 if CONFIG_SUPERNET['gpu_settings']['use_amp'] else 32,
                      amp_level='O1')

    trainer.fit(system)

