import argparse
import os
import numpy as np
import math
import itertools

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

# ten = tensor
# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, tensor, out=False, t = False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            tensor = self.conv(tensor)
            tensor_out = tensor
            tensor = self.bn(tensor)
            tensor = F.relu(tensor, False)
            return tensor, tensor_out
        else:
            tensor = self.conv(tensor)
            tensor = self.bn(tensor)
            tensor = F.relu(tensor, True)
            return tensor


# decoder block (used in the decoder)
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1,
                                       bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, tensor):
        tensor = self.conv(tensor)
        tensor = self.bn(tensor)
        tensor = F.relu(tensor, True)
        return tensor


class Encoder(nn.Module):
    def __init__(self, img_size:int=256, 
                       channel_in:int=32, 
                       target_size:int=8,
                       max_channel:int=256,
                       latent_dim:int=64):
        super().__init__()
        repeat_num = int(np.log2(img_size) - np.log2(target_size)) - 1
        blocks = []
        # the first time 3->64, for every other double the channel size
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        blocks += [nn.Conv2d(3, channel_in, 5, 1, 2)]
        blocks += [nn.BatchNorm2d(channel_in)]
        blocks += [nn.LeakyReLU(0.2)]
        for _ in range(repeat_num):
            channel_out = min(2*channel_in, max)
            blocks += [EncoderBlock(channel_in=channel_in, channel_out=channel_out)]
            channel_in = channel_out
        # final shape Bx256x8x8
        blocks += [nn.Conv2d(channel_out, channel_out, 3, 2, 1)]
        blocks += [nn.BatchNorm2d(channel_out)]
        blocks += [nn.LeakyReLU(0.2)]

        
        self.main = nn.Sequential(*blocks)
        self.fc = nn.Sequential(nn.Linear(in_features=target_size * target_size * channel_out, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.LeakyReLU(0.2))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1024, out_features=latent_dim)
        self.l_var = nn.Linear(in_features=1024, out_features=latent_dim)

    def forward(self, ten):
        ten = self.main(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, img_size:int=256, 
                       channel_in:int=256, 
                       start_size:int=8,
                       min_channel:int=16,
                       latent_dim:int=64):
        super(Decoder, self).__init__()
        self.start_size = start_size
        # start from B*latent_dim
        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=start_size * start_size * channel_in, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * channel_in, momentum=0.9),
                                nn.ReLU(True))
        repeat_num = int(np.log2(img_size) - np.log2(start_size)) - 1
        blocks = []
        blocks += [DecoderBlock(channel_in=channel_in, channel_out=channel_in)]
        for _ in range(repeat_num):
            channel_out = max(channel_in//2, min_channel)
            blocks += [DecoderBlock(channel_in=channel_in, channel_out=channel_out)]
            channel_in = channel_out

        # final conv to get 3 channels and tanh layer
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*blocks)

    def forward(self, ten):

        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, self.start_size, self.start_size)
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


class Discriminator(nn.Module):
    def __init__(self, channel_in=3,recon_level=3):
        super(Discriminator, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)))
        self.size = 32
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512,momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),

        )

    def forward(self, ten,other_ten,mode='REC'):
        if mode == "REC":
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                # we take the 9th layer as one of the outputs
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                    ten = lay(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            return F.sigmoid(ten)


    def __call__(self, *args, **kwargs):
        return super(Discriminator, self).__call__(*args, **kwargs)

class VaeGan(nn.Module):
    def __init__(self,latent_dim=50,recon_level=3):
        super(VaeGan, self).__init__()
        # latent space size
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder(latent_dim=self.latent_dim, size=self.encoder.size)
        self.discriminator = Discriminator(channel_in=3,recon_level=recon_level)
        # self-defined function to init the parameters
        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    #init as original implementatinumpy as np
                    scale = 1.0/np.sqrt(np.prod(m.weight.shape[1:]numpy as np
                    scale /=np.sqrt(3)
                    #nn.init.xavier_normal(m.weight,1)
                    #nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight,-scale,scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, ten, gen_size=10):
        if self.training:
            # save the original images
            ten_original = ten
            # encode
            mus, log_variances = self.encoder(ten)
            # we need the true variances, not the log one
            variances = torch.exp(log_variances * 0.5)
            # sample from a gaussian

            ten_from_normal = Variable(torch.randn(len(ten), self.latent_dim).cuda(), requires_grad=True)
            # shift and scale using the means and variances

            ten = ten_from_normal * variances + mus
            # decode the tensor
            ten = self.decoder(ten)
            # discriminator for reconstruction
            ten_layer = self.discriminator(ten, ten_original, "REC")
            # decoder for samples

            ten_from_normal = Variable(torch.randn(len(ten), self.latent_dim).cuda(), requires_grad=True)

            ten = self.decoder(ten_from_normal)
            ten_class = self.discriminator(ten_original, ten, "GAN")
            return ten, ten_class, ten_layer, mus, log_variances
        else:
            if ten is None:
                # just sample and decode

                ten = Variable(torch.randn(gen_size, self.latent_dim).cuda(), requires_grad=False)
                ten = self.decoder(ten)
            else:
                mus, log_variances = self.encoder(ten)
                # we need the true variances, not the log one
                variances = torch.exp(log_variances * 0.5)
                # sample from a gaussian

                ten_from_normal = Variable(torch.randn(len(ten), self.latent_dim).cuda(), requires_grad=False)
                # shift and scale using the means and variances
                ten = ten_from_normal * variances + mus
                # decode the tensor
                ten = self.decoder(ten)
            return ten



    def __call__(self, *args, **kwargs):
        return super(VaeGan, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(ten_original, ten_predict, layer_original, layer_predicted, labels_original,
             labels_sampled, mus, variances):
        """
        :param ten_original: original images
        :param ten_predict:  predicted images (output of the decoder)
        :param layer_original:  intermediate layer for original (intermediate output of the discriminator)
        :param layer_predicted: intermediate layer for reconstructed (intermediate output of the discriminator)
        :param labels_original: labels for original (output of the discriminator)
        :param labels_predicted: labels for reconstructed (output of the discriminator)
        :param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
        :param mus: tensor of means
        :param variances: tensor of diagonals of log_variances
        :return:
        """

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5*(ten_original.view(len(ten_original), -1) - ten_predict.view(len(ten_predict), -1)) ** 2
        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus,2) + variances + 1, 1)
        # mse between intermediate layers
        mse = torch.sum(0.5*(layer_original - layer_predicted) ** 2, 1)
        # bce for decoder and discriminator for original,sampled and reconstructed
        # the only excluded is the bce_gen_original

        bce_dis_original = -torch.log(labels_original + 1e-3)
        bce_dis_sampled = -torch.log(1 - labels_sampled + 1e-3)

        bce_gen_original = -torch.log(1-labels_original + 1e-3)
        bce_gen_sampled = -torch.log(labels_sampled + 1e-3)
        '''
        
        bce_gen_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.ones_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_gen_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
        bce_dis_original = nn.BCEWithLogitsLoss(size_average=False)(labels_original,
                                        Variable(torch.ones_like(labels_original.data).cuda(), requires_grad=False))
        bce_dis_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.zeros_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_dis_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.zeros_like(labels_sampled.data).cuda(), requires_grad=False))
        '''
        return nle, kl, mse, bce_dis_original, bce_dis_sampled,bce_gen_original,bce_gen_sampled