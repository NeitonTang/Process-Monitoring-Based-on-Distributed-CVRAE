
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.nn.modules.module import Module
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math

from itertools import product


class SVDD(nn.Module):
    """ VAE super class to reconstruct an image. Contains reparametrization
    method for latent variable z
    """
    def __init__(self, x_dim, z_in_dim, hidden_dim, z_dim, lamda=1e-5, device= "cuda"):
        super(SVDD, self).__init__()

        self.x_dim = x_dim
        self.h_dim = hidden_dim
        self.z_dim = z_dim
        self.z_in_dim = z_in_dim
        self.device = device#torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.n_unit = x_dim.shape[0]-1
        self.c = None
        self.R = None
        self.lamda = lamda
        self.MLP = nn.Sequential(nn.Linear(self.z_in_dim * self.n_unit, self.h_dim, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(self.h_dim, self.h_dim, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(self.h_dim, self.z_dim, bias=False))

    def forward(self, x_in):
        self.z = self.MLP(x_in)


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def loss_criterion(self):
        error_loss = torch.mean(torch.sum((self.z-self.c)**2, dim=2))
        penlty_loss = (torch.sum(self.MLP[0].weight**2)+torch.sum(self.MLP[2].weight**2)+torch.sum(self.MLP[4].weight**2)).view(1)
        return error_loss + self.lamda* penlty_loss

    def init_center_c(self, x_in,finish_flag =False, eps=0.1):
        if x_in is None:
            self.c /=self.n_samples
            self.c[(abs(self.c)<eps) & (self.c<0)] = -eps
            self.c[(abs(self.c)<eps) & (self.c>0)] = eps
            return
        if self.c is None:
            self.c=torch.zeros(self.z_dim, device=self.device)
            self.n_samples = 0
        outputs = self.MLP(x_in)
        self.c+=torch.sum(torch.sum(outputs,dim=0),dim=0)
        self.n_samples += outputs.shape[0] * outputs.shape[1]

class Autoencoder(nn.Module):
    def __init__(self, x_dim, z_in_dim, hidden_dim, z_dim, device= "cuda"):
        super(Autoencoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = hidden_dim
        self.z_dim = z_dim
        self.z_in_dim = z_in_dim
        self.device = device#torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.n_unit = x_dim.shape[0]-1
        self.c = None
        self.MLP = nn.Sequential(nn.Linear(self.z_in_dim * self.n_unit, self.h_dim, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(self.h_dim, self.h_dim, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(self.h_dim, self.z_dim, bias=False))

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, self.h_dim, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(self.h_dim, self.h_dim, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(self.h_dim, self.z_in_dim * self.n_unit, bias=False))

    def forward(self, x_in):
        self.x_in = x_in
        self.z = self.MLP(x_in)
        self.x_re = self.decoder(self.z)

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def loss_criterion(self):
        return torch.mean(torch.sum((self.x_in-self.x_re)**2, dim=2))