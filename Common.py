"""This is used to:
   1.get the Train and val Datast,
   considering the train augument later
   2.get the optimizer
   """

from dataset.Get_Dataset import GetDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.tools import make_dirs
import torch
import sys
import numpy as np
from model.VoxResNet import VoxResNet_3D
from dataset.joint_transform import *
from config import opt

all_mean1 = np.array([168.5], dtype=np.float32)
all_std1 = np.array([500], dtype=np.float32)

model_choice={
    'VoxResNet_3D':VoxResNet_3D,
}

# decide to use which Model: currently UNet
def GetModel(opt):
    if opt.model_choice == 'VoxResNet_3D':
        model=model_choice[opt.model_choice](opt.in_dim,opt.out_dim)

    # decide whether to use cuda or not
    if opt.use_cuda:
        model=torch.nn.DataParallel(model).cuda()
    return model


def GetDatasetLoader(opt,phase= 'train'):
    # decide whether to augument or not
    if phase == 'train':
        # set the data augumentation for the training image
        train_aug = JointCompose([JointRandomFlip(),
                                  JointRandomRotation(),
                                  JointRandomGaussianNoise(10, all_std1),
                                  JointRandomSubTractGaussianNoise(10, all_std1),
                                  JointRandomBrightness([-0.5, 0.5]),
                                  JointRandomIntensityChange([-100, 100], all_std1),
                                  ])
        dataset1 = GetDataset(opt.dataset_prefix, phase, augument=train_aug)
        dataloader1 = DataLoader(dataset1, batch_size=opt.train_batch, num_workers=opt.num_workers,
                                 shuffle=opt.train_shuffile)

    if phase == 'val':
        dataset1=GetDataset(opt.dataset_prefix,phase,augument=None)
        dataloader1 = DataLoader(dataset1, batch_size=opt.val_batch, num_workers=opt.num_workers, shuffle=False)

    if phase == 'test':
        dataset1 = GetDataset(opt.dataset_prefix,phase,augument=None)
        dataloader1 = DataLoader(dataset1, batch_size=opt.test_batch, num_workers=opt.num_workers, shuffle=False)

    return dataloader1


def GetOptimizer(opt,model):
    if opt.optimizer=='SGD':
        optimizer1=optim.SGD(
            model.parameters(),lr=opt.lr,momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )

    if opt.optimizer=='RMSp':
        optimizer1=optim.RMSprop(
            model.parameters(),lr=opt.lr,alpha=opt.alpha,
            weight_decay=opt.weight_decay,
        )

    return optimizer1


def GetScheduler(opt,optimizer1):
    scheduler1=optim.lr_scheduler.StepLR(
        optimizer1,step_size=opt.step_size,gamma=opt.gamma
    )
    return scheduler1


def adjust_lr(opt,optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# save the result
def save_parameters(state,best_value='UNet'):
    save_path = 'checkpoints'
    make_dirs(save_path)

    # save_name = save_path + '/model_parameters_value{:.3f}.pth'.format(best_value)
    save_name = save_path + '/{}.ckpt'.format(best_value)
    torch.save(state, save_name)

def load_checkpoint(model, checkpoint_PATH, optimizer):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    optimizer.load_state_dict(model_CKPT['optimizer'])
    start_epoch = model_CKPT['epoch'] + 1

    return model, optimizer,start_epoch


def GetWeight(opt,target,slr=0.0001,is_t=1):
    if target.device.type=='cuda':
        beta = target.sum().cpu().numpy().astype(np.float32) / (target.numel() + 1e-5)
    else:
        beta = target.sum().numpy().astype(np.float32) / (target.numel() + 1e-5)

    beta = beta + slr
    weight = np.array([beta, 1 - beta])

    if is_t:
        weight = torch.tensor(weight)
        if opt.use_cuda:
            weight = weight.float().cuda()

    return weight


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass



#### write the log files
def WriteLog(log_path,log_name,opt1):
    log = Logger()
    make_dirs(log_path)
    log.open(log_name, mode='a')
    log.write('** experiment settings **\n')
    log.write('\toptimizer:               {}  \n'.format(opt1.optimizer))
    log.write('\tlearning rate:         {:.3f}\n'.format(opt1.lr))
    log.write('\tweight_decay:         {:.4f}\n'.format(opt1.weight_decay))
    log.write('\tepoches:               {:.3f}\n'.format(opt1.train_epoch))
    log.write('\tpatch_size:              {}  \n'.format(opt1.patch_size))
    log.write('\tmodel:                   {}  \n'.format(opt1.model_choice))
    log.write('\tsave_parameter:                   {}  \n'.format(opt1.save_parameters_name))

    return log


