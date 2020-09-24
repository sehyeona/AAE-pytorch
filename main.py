import torch
import numpy
import argparse

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

from network import VaeGan
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from torch.optim import RMSprop,Adam,SGD
from torch.optim.lr_scheduler import ExponentialLR,MultiStepLR

import progressbar
from torchvision.utils import make_grid

from utils import RollingMeasure
from core.dataloader import get_train_loader

if __name__ == '__main__':
    


