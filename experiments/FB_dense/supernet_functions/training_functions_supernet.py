import torch
import time
from general_functions.utils import AverageMeter, save
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
import numpy as np

class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer):
        self.logger = logger
        self.writer = writer
        
        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler
        
        self.temperature                 = CONFIG_SUPERNET['train_settings']['init_temperature']
        self.exp_anneal_rate             = CONFIG_SUPERNET['train_settings']['exp_anneal_rate'] # apply it every epoch
        self.cnt_epochs                  = CONFIG_SUPERNET['train_settings']['cnt_epochs']
        self.train_thetas_from_the_epoch = CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch']
        self.print_freq                  = CONFIG_SUPERNET['train_settings']['print_freq']
        self.path_to_save_model          = CONFIG_SUPERNET['train_settings']['path_to_save_model']
        self.best_mse = 99999
        self.best_mae = 99999
        self.device = "cuda"

    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):
        # firstly, train weights only
        for epoch in range(self.train_thetas_from_the_epoch):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            self.logger.info("Firstly, start to train weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")

        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('learning_rate/theta', self.theta_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Start to train weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            self._validate(model, test_loader, epoch)
            self.w_scheduler.step()

    def _training_step(self, model, loader, optimizer, epoch, info_for_logger=""):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        start_time = time.time()

        model = model.train()
        epoch_length = len(loader)
        for step, (inputs, points, targets, st_sizes) in enumerate(loader):
            if targets is None:
                continue
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]
            if info_for_logger == "_theta_step_" and step > epoch_length/3:
                continue
            with torch.set_grad_enabled(True):
                outs = model(inputs, self.temperature)
                loss = self.criterion(outs, points, targets, st_sizes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                N = inputs.shape[0]
                pre_count = torch.sum(outs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
            if step % self.print_freq ==0:
                self.logger.info('Epoch {} Step {} Train, Loss: {:.2f}/({:.2f}), MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(epoch, step, epoch_loss.get_avg(),loss.item(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(), time.time()-start_time))

        self.writer.add_scalar('train_vs_val/' + "train" + '_loss' + info_for_logger, epoch_loss.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/' + "train" + '_mse' + info_for_logger, np.sqrt(epoch_mse.get_avg()), epoch)
        self.writer.add_scalar('train_vs_val/' + "train" + '_mae' + info_for_logger, epoch_mae.get_avg(), epoch)

    def _validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()
        epoch_res = []
        with torch.no_grad():
            for step, (inputs, count, name) in enumerate(loader):
                inputs = inputs.cuda()
                # inputs are images with different sizes
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
                with torch.set_grad_enabled(False):
                    outs = model(inputs, self.temperature)
                    res = count[0].item() - torch.sum(outs).item()
                    epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(epoch, mse, mae, time.time()-start_time))
        self.writer.add_scalar('train_vs_val/' + "val" + '_mse' + '', mse, epoch)
        self.writer.add_scalar('train_vs_val/' + "val" + '_mae' + '', mae, epoch)

        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 epoch))
            torch.save(model.state_dict(), self.path_to_save_model)
