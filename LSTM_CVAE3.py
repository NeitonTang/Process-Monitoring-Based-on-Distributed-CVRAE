import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math

from itertools import product


class LSTMCVAE(nn.Module):
    """ VAE super class to reconstruct an image. Contains reparametrization
    method for latent variable z
    """

    def __init__(self, x_dim, c_dim, hidden_dim, z_dim, cz_dim, M_sample=30, device=torch.device("cuda:0")):
        super(LSTMCVAE, self).__init__()

        self.x_dim = x_dim
        self.c_dim = c_dim
        self.h_dim = hidden_dim
        self.z_dim = z_dim
        self.cz_dim = cz_dim
        self.M_sample = M_sample
        self.device = device  # torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        # features c
        self.c_mlp = nn.LSTMCell(self.c_dim, self.cz_dim)

        # Generator P(x|z, c)
        self.pxgzc_mlp1 = nn.Sequential(
            nn.Linear(self.z_dim + self.cz_dim, self.h_dim),
            nn.ReLU())
        self.pxgzc_mean = nn.Linear(self.h_dim, self.x_dim)
        self.pxgzc_log_std2 = nn.Linear(self.h_dim, self.x_dim)

        # Recogniser Q(z|x, c)
        self.qzgxc_mlp1 = nn.LSTMCell(self.x_dim + self.cz_dim, self.h_dim)
        self.qzgxc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.qzgxc_log_std2 = nn.Linear(self.h_dim, self.z_dim)

        # Recogniser p(z|c)
        self.pzgc_mlp = nn.Sequential(
            nn.Linear(self.cz_dim, self.h_dim),
            nn.ReLU())
        self.pzgc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.pzgc_log_std2 = nn.Linear(self.h_dim, self.z_dim)

    def Encoder(self, x, c):
        batch_num = x.size(0)
        self.x = torch.reshape(x, (-1, self.x_dim))
        self.c = torch.reshape(c, (-1, self.c_dim))
        self.h_tc, self.c_tc = self.c_mlp(self.c, (self.h_tc, self.c_tc))
        self.h_t0, self.c_t0 = self.qzgxc_mlp1(torch.cat((self.x, self.h_tc), 1), (self.h_t0, self.c_t0))
        self.z_mu = self.qzgxc_mean(self.h_t0)
        self.z_log_var = self.qzgxc_log_std2(self.h_t0)

    def Decoder(self):
        self.h_t2 = self.pxgzc_mlp1(torch.cat((self.z, self.h_tc.unsqueeze(0).repeat((self.M_sample, 1,1))), 2))
        self.x_mu = self.pxgzc_mean(self.h_t2)
        self.x_log_var = self.pxgzc_log_std2(self.h_t2)

        self.h_pzgc = self.pzgc_mlp(self.h_tc)
        self.z_pior_mu = self.pzgc_mean(self.h_pzgc)
        self.z_pior_log_var = self.pzgc_log_std2(self.h_pzgc)
        # return self.z_mu, self.z_log_var, x_mu, x_log_var

    def x_sample(self, x_mu, x_log_var):
        epsilon = Variable(torch.randn(1, self.x_dim),
                           requires_grad=False).to(self.device)
        x = epsilon * torch.exp(x_log_var / 2) + x_mu  # 2 for convert var to std
        return x[0, :, :]

    def reparameterize(self):
        """" Reparametrization trick: z = mean + std*epsilon,
        where epsilon ~ N(0, 1).
        """
        epsilon = Variable(torch.FloatTensor(self.z_mu.size(0), self.z_mu.size(1)).normal_(), requires_grad=False).to(
            self.device)
        self.z = epsilon * torch.exp(self.z_log_var / 2) + self.z_mu  # 2 for convert var to std
        # return z

    def kld_Term(self, mean_1, log_std2_1, mean_2, log_std2_2):
        kld_Celement = 0.5 * (
                -log_std2_1 + log_std2_2 + ((- mean_1 + mean_2).pow(2) + torch.exp(log_std2_1)) / torch.exp(
            log_std2_2) - 1)
        return kld_Celement

    def kl_divergence(self):
        """ Compute Kullback-Leibler divergence """
        self.kl_loss = torch.mean(
            torch.sum(self.kld_Term(self.z_mu, self.z_log_var, self.z_pior_mu, self.z_pior_log_var), 1))

    def recon_lossTerm(self):
        self.recon_loss = torch.mean(torch.sum(-0.5 * (math.log(2 * math.pi) + self.x_log_var
                                                       + (self.x_mu - self.x) ** 2 / torch.exp(self.x_log_var)), 2))

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def loss_criterion(self):
        self.kl_divergence()
        self.recon_lossTerm()
        return self.recon_loss - 0.8 * self.kl_loss

    def x_error(self):
        return torch.mean(torch.sum((self.x_mu - self.x) ** 2, 2))