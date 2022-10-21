
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

from LSTM_CVAE3 import LSTMCVAE


class VAEModel(nn.Module):
    """ VAE super class to reconstruct an image. Contains reparametrization
    method for latent variable z
    """
    def __init__(self, x_dim, y_dim,  hidden_dim, z_dim, cz_dim=5, M_sample = 30, device= torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")):
        super(VAEModel, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = hidden_dim
        self.z_dim = z_dim
        self.cz_dim = cz_dim
        self.M_sample = M_sample
        self.num_layers = 2
        self.device = device#torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        #self.c_dim = np.ones(x_dim.shape[0] - 1) * hidden_dim
        #self.c_dim[0] = x_dim[0]
        #sub_vae = nn.ModuleList(
        #   LSTMCVAE(x_dim[i + 1], c_dim[i], hidden_dim, z_dim, M_sample) for i in range(x_dim.shapep[0] - 1))
        self.sub_vae1 = LSTMCVAE(x_dim[1], x_dim[0], hidden_dim, z_dim, cz_dim, M_sample).to(device)
        self.sub_vae2 = LSTMCVAE(x_dim[2], hidden_dim, hidden_dim, z_dim, cz_dim, M_sample).to(device)
        self.sub_vae3 = LSTMCVAE(x_dim[3], hidden_dim, hidden_dim, z_dim, cz_dim, M_sample).to(device)
        self.sub_vae4 = LSTMCVAE(x_dim[4], hidden_dim, hidden_dim, z_dim, cz_dim, M_sample).to(device)
        self.sub_vae5 = LSTMCVAE(x_dim[5], hidden_dim, hidden_dim, z_dim, cz_dim, M_sample).to(device)
        self.sub_vae6 = LSTMCVAE(x_dim[6], hidden_dim, hidden_dim, z_dim, cz_dim, M_sample).to(device)
        self.sub_vae7 = LSTMCVAE(x_dim[7], hidden_dim, hidden_dim, z_dim, cz_dim, M_sample).to(device)
        # p(y|h)
        self.pygh_mpl1 = nn.Sequential(
            nn.Linear(self.z_dim + self.cz_dim, self.h_dim),
            nn.ReLU())
        self.pygh_mean = nn.Linear(self.h_dim, self.y_dim)
        self.pygh_log_std2 = nn.Linear(self.h_dim, self.y_dim)

    def forward(self, x, y):
        batch_num = x.size(0)
        k_step = x.size(1)
        self.init_hstate(batch_num)
        loss =[]
        self.z_to = []
        self.z_normal = []
        for i in range(k_step):
            self.y = y[:, i, :].reshape(-1, self.y_dim)
            self.sub_vae1.Encoder(x[:, i, np.sum(self.x_dim[0:1]):np.sum(self.x_dim[0:1]) + self.x_dim[1]],
                                  x[:, i, 0:self.x_dim[0]])
            self.reparameterize(self.sub_vae1)
            self.sub_vae1.Decoder()

            self.sub_vae2.Encoder(x[:, i, np.sum(self.x_dim[0:2]):np.sum(self.x_dim[0:2]) + self.x_dim[2]],
                                  self.sub_vae1.h_t0)
            self.reparameterize(self.sub_vae2)
            self.sub_vae2.Decoder()

            self.sub_vae3.Encoder(x[:, i, np.sum(self.x_dim[0:3]):np.sum(self.x_dim[0:3]) + self.x_dim[3]],
                                  self.sub_vae2.h_t0)
            self.reparameterize(self.sub_vae3)
            self.sub_vae3.Decoder()

            self.sub_vae4.Encoder(x[:, i, np.sum(self.x_dim[0:4]):np.sum(self.x_dim[0:4]) + self.x_dim[4]],
                                  self.sub_vae3.h_t0)
            self.reparameterize(self.sub_vae4)
            self.sub_vae4.Decoder()

            self.sub_vae5.Encoder(x[:, i, np.sum(self.x_dim[0:5]):np.sum(self.x_dim[0:5]) + self.x_dim[5]],
                                  self.sub_vae4.h_t0)
            self.reparameterize(self.sub_vae5)
            self.sub_vae5.Decoder()

            self.sub_vae6.Encoder(x[:, i, np.sum(self.x_dim[0:6]):np.sum(self.x_dim[0:6]) + self.x_dim[6]],
                                  self.sub_vae5.h_t0)
            self.reparameterize(self.sub_vae6)
            self.sub_vae6.Decoder()

            self.sub_vae7.Encoder(x[:, i, np.sum(self.x_dim[0:7]):np.sum(self.x_dim[0:7]) + self.x_dim[7]],
                                  self.sub_vae6.h_t0)
            self.reparameterize(self.sub_vae7)
            self.sub_vae7.Decoder()
            self.z_to.append((torch.cat([self.sub_vae1.z, self.sub_vae2.z, self.sub_vae3.z, self.sub_vae4.z,
                                         self.sub_vae5.z, self.sub_vae6.z, self.sub_vae7.z], dim=2)).unsqueeze(2))
            self.z_normal.append((torch.cat([(self.sub_vae1.z-self.sub_vae1.z_pior_mu)/torch.exp(self.sub_vae1.z_pior_log_var/2),
                                             (self.sub_vae2.z - self.sub_vae2.z_pior_mu) / torch.exp(self.sub_vae2.z_pior_log_var / 2),
                                             (self.sub_vae3.z - self.sub_vae3.z_pior_mu) / torch.exp(self.sub_vae3.z_pior_log_var / 2),
                                             (self.sub_vae4.z - self.sub_vae4.z_pior_mu) / torch.exp(self.sub_vae4.z_pior_log_var / 2),
                                             (self.sub_vae5.z - self.sub_vae5.z_pior_mu) / torch.exp(self.sub_vae5.z_pior_log_var / 2),
                                             (self.sub_vae6.z - self.sub_vae6.z_pior_mu) / torch.exp(self.sub_vae6.z_pior_log_var / 2),
                                             (self.sub_vae7.z - self.sub_vae7.z_pior_mu) / torch.exp(self.sub_vae7.z_pior_log_var / 2)], dim=2)).unsqueeze(2))
            self.h_t0 = self.pygh_mpl1(torch.cat((self.sub_vae7.z, self.sub_vae7.h_tc.unsqueeze(0).repeat((self.M_sample, 1,1))), 2))
            self.y_mu = self.pygh_mean(self.h_t0)
            self.y_log_var = self.pygh_log_std2(self.h_t0)
            if i > 20:
                loss.append(self.loss_criterion())
        return torch.mean(torch.stack(loss))

    def init_hstate(self, batch_size=1):
        self.sub_init_hstate(self.sub_vae1, batch_size)
        self.sub_init_hstate(self.sub_vae2, batch_size)
        self.sub_init_hstate(self.sub_vae3, batch_size)
        self.sub_init_hstate(self.sub_vae4, batch_size)
        self.sub_init_hstate(self.sub_vae5, batch_size)
        self.sub_init_hstate(self.sub_vae6, batch_size)
        self.sub_init_hstate(self.sub_vae7, batch_size)

    def sub_init_hstate(self, sub_vae, batch_size=1):
        sub_vae.h_tc = Variable(torch.FloatTensor(batch_size, sub_vae.cz_dim).normal_(), requires_grad=True).to(self.device)
        sub_vae.c_tc = Variable(torch.FloatTensor(batch_size, sub_vae.cz_dim).normal_(), requires_grad=True).to(self.device)
        sub_vae.h_t0 = Variable(torch.FloatTensor(batch_size, sub_vae.h_dim).normal_(), requires_grad=True).to(self.device)
        sub_vae.c_t0 = Variable(torch.FloatTensor(batch_size, sub_vae.h_dim).normal_(), requires_grad=True).to(self.device)

    def reparameterize(self, sub_vae):
        epsilon = Variable(torch.FloatTensor(sub_vae.M_sample, sub_vae.z_mu.size(0), sub_vae.z_mu.size(1)).normal_(), requires_grad=False).to(self.device)
        sub_vae.z = epsilon * torch.exp(sub_vae.z_log_var / 2) + sub_vae.z_mu# 2 for convert var to std

    def recon_lossTerm_y(self):
        self.recon_loss_y = torch.mean(torch.sum(-0.5*(math.log(2*math.pi) + self.y_log_var +
                                                       (self.y_mu - self.y) ** 2 / torch.exp(self.y_log_var)),2))

    def y_error(self):
        return torch.mean(torch.sum((self.y_mu-self.y) ** 2, 2))

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
        self.sub_vae1.reset_parameters()
        self.sub_vae2.reset_parameters()
        self.sub_vae3.reset_parameters()
        self.sub_vae4.reset_parameters()
        self.sub_vae5.reset_parameters()
        self.sub_vae6.reset_parameters()
        self.sub_vae7.reset_parameters()


    def loss_criterion(self):
        self.sub1_loss = self.sub_vae1.loss_criterion()
        self.sub2_loss = self.sub_vae2.loss_criterion()
        self.sub3_loss = self.sub_vae3.loss_criterion()
        self.sub4_loss = self.sub_vae4.loss_criterion()
        self.sub5_loss = self.sub_vae5.loss_criterion()
        self.sub6_loss = self.sub_vae6.loss_criterion()
        self.sub7_loss = self.sub_vae7.loss_criterion()
        self.recon_lossTerm_y()
        return -(self.recon_loss_y + self.sub1_loss
                 + self.sub2_loss + self.sub3_loss
                 + self.sub4_loss + self.sub5_loss
                 + self.sub6_loss + self.sub7_loss)

    def SetOptimizer(self):
        train_flag = [True, True, True, False, False, False, False]
        for p in self.sub_vae1.parameters():
            p.requires_grad = train_flag[0]
        for p in self.sub_vae2.parameters():
            p.requires_grad = train_flag[1]
        for p in self.sub_vae3.parameters():
            p.requires_grad = train_flag[2]
        for p in self.sub_vae4.parameters():
            p.requires_grad = train_flag[3]
        for p in self.sub_vae5.parameters():
            p.requires_grad = train_flag[4]
        for p in self.sub_vae6.parameters():
            p.requires_grad = train_flag[5]
        for p in self.sub_vae7.parameters():
            p.requires_grad = train_flag[6]


    def seq_out(self, x):
        self.sub_vae1.Encoder(x[np.sum(self.x_dim[0:1]):np.sum(self.x_dim[0:1]) + self.x_dim[1]],
                              x[0:self.x_dim[0]])
        self.reparameterize(self.sub_vae1)
        self.sub_vae1.Decoder()
        self.sub_vae2.Encoder(x[np.sum(self.x_dim[0:2]):np.sum(self.x_dim[0:2]) + self.x_dim[2]],
                              self.sub_vae1.h_t0)
        self.reparameterize(self.sub_vae2)
        self.sub_vae2.Decoder()
        self.sub_vae3.Encoder(x[np.sum(self.x_dim[0:3]):np.sum(self.x_dim[0:3]) + self.x_dim[3]],
                              self.sub_vae2.h_t0)
        self.reparameterize(self.sub_vae3)
        self.sub_vae3.Decoder()
        self.sub_vae4.Encoder(x[np.sum(self.x_dim[0:4]):np.sum(self.x_dim[0:4]) + self.x_dim[4]],
                              self.sub_vae3.h_t0)
        self.reparameterize(self.sub_vae4)
        self.sub_vae4.Decoder()
        self.sub_vae5.Encoder(x[np.sum(self.x_dim[0:5]):np.sum(self.x_dim[0:5]) + self.x_dim[5]],
                              self.sub_vae4.h_t0)
        self.reparameterize(self.sub_vae5)
        self.sub_vae5.Decoder()
        self.sub_vae6.Encoder(x[np.sum(self.x_dim[0:6]):np.sum(self.x_dim[0:6]) + self.x_dim[6]],
                              self.sub_vae5.h_t0)
        self.reparameterize(self.sub_vae6)
        self.sub_vae6.Decoder()
        self.sub_vae7.Encoder(x[np.sum(self.x_dim[0:7]):np.sum(self.x_dim[0:7]) + self.x_dim[7]],
                              self.sub_vae6.h_t0)
        self.reparameterize(self.sub_vae7)
        self.sub_vae7.Decoder()
        self.h_t0= self.pygh_mpl1(
            torch.cat((self.sub_vae7.z, self.sub_vae7.h_tc.repeat((self.M_sample, 1, 1))), 2))
        self.y_mu = self.pygh_mean(self.h_t0)
        self.y_log_var = self.pygh_log_std2(self.h_t0)
        self.x_mu = [self.sub_vae1.x_mu, self.sub_vae2.x_mu, self.sub_vae3.x_mu, self.sub_vae4.x_mu, self.sub_vae5.x_mu, self.sub_vae6.x_mu, self.sub_vae7.x_mu]
        self.x_log_std = [self.sub_vae1.x_log_var, self.sub_vae2.x_log_var, self.sub_vae3.x_log_var, self.sub_vae4.x_log_var, self.sub_vae5.x_log_var,
                     self.sub_vae6.x_log_var, self.sub_vae7.x_log_var]
        self.z_mu = [self.sub_vae1.z_mu, self.sub_vae2.z_mu, self.sub_vae3.z_mu, self.sub_vae4.z_mu, self.sub_vae5.z_mu,
                self.sub_vae6.z_mu, self.sub_vae7.z_mu]
        self.z = [self.sub_vae1.z, self.sub_vae2.z, self.sub_vae3.z, self.sub_vae4.z, self.sub_vae5.z,
                     self.sub_vae6.z, self.sub_vae7.z]
        self.z_log_std = [self.sub_vae1.z_log_var, self.sub_vae2.z_log_var, self.sub_vae3.z_log_var, self.sub_vae4.z_log_var,
                     self.sub_vae5.z_log_var, self.sub_vae6.z_log_var, self.sub_vae7.z_log_var]
        self.z_pior_mu = [self.sub_vae1.z_pior_mu, self.sub_vae2.z_pior_mu, self.sub_vae3.z_pior_mu, self.sub_vae4.z_pior_mu, self.sub_vae5.z_pior_mu,
                     self.sub_vae6.z_pior_mu, self.sub_vae7.z_pior_mu]
        self.z_pior_log_std = [self.sub_vae1.z_pior_log_var, self.sub_vae2.z_pior_log_var, self.sub_vae3.z_pior_log_var, self.sub_vae4.z_pior_log_var,
                          self.sub_vae5.z_pior_log_var, self.sub_vae6.z_pior_log_var, self.sub_vae7.z_pior_log_var]
        self.x = [self.sub_vae1.x, self.sub_vae2.x, self.sub_vae3.x, self.sub_vae4.x, self.sub_vae5.x,
              self.sub_vae6.x, self.sub_vae7.x]
        return torch.mean(self.y_mu, 0).squeeze().to(torch.device("cpu")), torch.mean(torch.exp(self.y_log_var / 2), 0).squeeze().to(torch.device("cpu"))