import torch
from torch.autograd import Variable
import time
from general_functions.utils import AverageMeter, save, accuracy
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
    
    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):
        
        best_top1 = 0.0
        
        # firstly, train weights only
        for epoch in range(self.train_thetas_from_the_epoch):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Firstly, start to train weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            self.w_scheduler.step()
        
        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('learning_rate/theta', self.theta_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Start to train weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            self.w_scheduler.step()
            
            self.logger.info("Start to train theta for epoch %d" % (epoch))
            self._training_step(model, train_thetas_loader, self.theta_optimizer, epoch, info_for_logger="_theta_step_")
            
            top1_avg = self._validate(model, test_loader, epoch)
            if best_top1 < top1_avg:
                best_top1 = top1_avg
                self.logger.info("Best top1 acc by now. Save model")
                save(model, self.path_to_save_model)
            
            self.temperature = self.temperature * self.exp_anneal_rate
       
    def _training_step(self, model, loader, optimizer, epoch, info_for_logger=""):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()

        model = model.train()
        start_time = time.time()

        for step, (inputs, points, targets, st_sizes) in enumerate(loader):
            inputs = inputs.cuda()
            st_sizes = st_sizes.cuda()
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.cuda() for p in points]
            targets = [t.cuda() for t in targets]
            # import pdb;pdb.set_trace()
            N = inputs.shape[0]
            optimizer.zero_grad()
            latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).to(inputs.device)
            outs, latency_to_accumulate = model(inputs, self.temperature, latency_to_accumulate)
            loss = self.criterion(outs, points, targets, st_sizes, latency_to_accumulate)

            loss.backward()
            optimizer.step()
            pre_count = torch.sum(outs.view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            epoch_loss.update(loss.item(), N)
            epoch_mse.update(np.mean(res * res), N)
            epoch_mae.update(np.mean(abs(res)), N)
            if step % self.print_freq ==0:
                self.logger.info('Epoch {} Step {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(epoch, step, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                        time.time()-start_time))
        
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
                    latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).to(inputs.device)
                    outs, latency_to_accumulate = model(inputs, self.temperature, latency_to_accumulate)
                    res = count[0].item() - torch.sum(outs).item()
                    epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(epoch, mse, mae, time.time()-start_time))
        top1_avg = mse + mae
        return top1_avg