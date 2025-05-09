import os
import logging
import torch
import numpy as np

class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
  
    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)
  
    def get_avg(self):
        return self.avg
    
    def __repr__(self):
        return self.__str__()

def weights_init(m, deepth=0, max_depth=2):
    if deepth > max_depth:
        return
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.BatchNorm2d):
        return
    elif isinstance(m, torch.nn.ReLU):
        return
    elif isinstance(m, torch.nn.Module):
        deepth += 1
        for m_ in m.modules():
            weights_init(m_, deepth)
    else:
        raise ValueError("%s is unk" % m.__class__.__name__)

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('fbnet')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

# Example:
# utils.save(model, os.path.join(args.save, 'weights.pt'))
def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

def add_text_to_file(text, file_path):
    with open(file_path, 'a') as f:
        f.write(text)
    
def clear_files_in_the_list(list_of_paths):
    for file_name in list_of_paths:
        open(file_name, 'w').close()

def create_directories_from_list(list_of_directories):
    for directory in list_of_directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
       
def accuracy(output, target):
    output = output.to(output.device)
    epoch_minus = []
    assert output.size(0) == 1, 'the batch size should equal to 1'
    with torch.set_grad_enabled(False):
        temp_minu = torch.sum(target[0]).item() - torch.sum(output).item()
        epoch_minus.append(temp_minu)
    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    return mse, mae

def check_tensor_in_list(atensor, alist):
    if any([(atensor == t_).all() for t_ in alist if atensor.shape == t_.shape]):
        return True
    return False