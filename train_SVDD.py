import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
from VAE_Model3 import VAEModel
from SVDD import SVDD, Autoencoder


x_dim = np.array([5, 6, 8, 8, 8, 8, 8, 7], dtype=np.int16)
x_dim_sum = np.sum(x_dim).astype(int)
y_dim = 2
z_dim = 7
global_z_dim = 10
c_dim = 5
h_dim = 30
train_batch = 256
test_batch = 1000
k_step = 50
file_name = '../data/data_01.xls'
n_epochs = 10000
learning_rate = 1e-3
clip = 10
model_flag = 1
M_sample =30
nu = 0.01
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
# Decide which device we want to run on
device = torch.device("cuda:0")
#device = torch.device("cpu")

fn0 = './saves/LSTM_state_dict_3.pth'
state_dict = torch.load(fn0)
model0 = VAEModel(x_dim, y_dim, h_dim, z_dim, c_dim, M_sample).to(device)
model0.load_state_dict(state_dict)

torch.manual_seed(256)
np.random.seed(256)

def dataset_loader(filename, k_step, train_batch):
    file_data = np.array(pd.read_excel(filename, header=1))
    file_data = file_data[2000:7000, :]
    batch_random = np.random.permutation(len(file_data)-k_step+1)
    data_mean = file_data.mean(0)
    data_std = file_data.std(0)
    file_data = (file_data - data_mean)/data_std
    file_data[:, 0:x_dim[0]] = 1
    # 保存模型示例代码
    print('===> Saving data mean and std')
    state = {
        'mean': data_mean,
        'std': data_std
    }
    torch.save(state, './saves/mean_std.t7')

    train_data = []
    train_target = []
    for i in range(0, len(batch_random)):
        train_data.append((torch.from_numpy(
            np.reshape(file_data[batch_random[i]:batch_random[i] + k_step, 0:x_dim_sum], (1, k_step, x_dim_sum)))).to(
            dtype=torch.float))
        train_target.append(
            (torch.from_numpy(np.reshape(file_data[batch_random[i]:batch_random[i]+k_step, x_dim_sum:x_dim_sum+y_dim],
                                         (1, k_step, y_dim)))).to(dtype=torch.float))

    train_data = torch.cat(train_data, dim=0)
    train_target = torch.cat(train_target, dim=0)
    train_dataset = Data.TensorDataset(train_data, train_target)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch,
        shuffle=True)

    valid_data = []
    valid_target = []
    for i in range(len(batch_random)-200, len(batch_random)):
        valid_data.append((torch.from_numpy(
            np.reshape(file_data[batch_random[i]:batch_random[i] + k_step, 0:x_dim_sum], (1, k_step, x_dim_sum)))).to(
            dtype=torch.float))
        valid_target.append(
            (torch.from_numpy(
                np.reshape(file_data[batch_random[i]:batch_random[i] + k_step, x_dim_sum:x_dim_sum + y_dim],
                           (1, k_step, y_dim)))).to(dtype=torch.float))

    valid_data = torch.cat(valid_data, dim=0)
    valid_target = torch.cat(valid_target, dim=0)
    valid_dataset = Data.TensorDataset(valid_data, valid_target)
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=train_batch,
        shuffle=False)

    test_batch = file_data.shape[0]
    test_data = file_data[:, 0:x_dim_sum]
    test_target = file_data[:, x_dim_sum:x_dim_sum+y_dim]
    test_data = (torch.from_numpy(test_data)).to(dtype=torch.float)
    test_target = (torch.from_numpy(test_target)).to(dtype=torch.float)
    test_dataset = Data.TensorDataset(test_data, test_target)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch,
        shuffle=False)
    return train_loader, valid_loader, test_loader

def model_pretrain(permodel, dataset, optimizer, maxepoch=100):
    perfn = "./saves/pertrain_autoencoder_dict.pth"
    best_loss = 1e10
    last_step = 0
    for epoch in range(1, maxepoch + 1):
        print("pertrain epoch {}".format(epoch))
        train_losses = []
        permodel.train()  # prep model for training
        for n_batch, (input_data, targets) in enumerate(dataset):
            sys.stdout.write('\rEpoch: [%s--%s] ' % (epoch, n_batch))
            sys.stdout.flush()
            data = Variable(input_data).to(device)
            y = Variable(targets).to(device)
            with torch.no_grad():
                model0(data, y)

            optimizer.zero_grad()
            x_in = model0.z_normal
            x_in = torch.cat(x_in, dim=2)
            x_in = torch.cat([x_in[:, :, i, :] for i in range(20, k_step)], dim=1)
            x_in = Variable(x_in, requires_grad=False)
            permodel(x_in)
            loss = permodel.loss_criterion()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # grad norm clipping, only in pytorch version >= 1.10
            nn.utils.clip_grad_norm_(permodel.parameters(), clip)
            # record training loss
            train_losses.append(loss.item())

        train_loss = np.average(train_losses)
        if train_loss<best_loss:
            best_loss=train_loss
            torch.save(permodel.state_dict(), perfn)
            last_step = 0
        else:
            last_step +=1
        print_msg = ("pertrain epoch:{}   ".format(epoch) + "train_loss: {:.5f} ".format(train_loss)
                     + "best_loss: {:.5f}".format(best_loss) +  "   last_step: {}".format(last_step))
        print(print_msg)

        if last_step>20 or epoch>=maxepoch:
            permodel.load_state_dict(torch.load(perfn))
            return

def init_network_weights_from_pretraining(model, permodel):
    net_dict = model.state_dict()
    ae_net_dict = permodel.state_dict()

    # Filter out decoder network keys
    ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
    # Overwrite values in the existing state_dict
    net_dict.update(ae_net_dict)
    # Load the new state_dict
    model.load_state_dict(net_dict)

def init_center_c(model, dataset):
    model.eval()
    with torch.no_grad():
        model.c = None
        for n_batch, (input_data, targets) in enumerate(dataset):
            data = Variable(input_data).to(device)
            y = Variable(targets).to(device)
            model0(data, y)
            x_in = model0.z_normal
            x_in = torch.cat(x_in, dim=2)
            x_in = torch.cat([x_in[:, :, i, :] for i in range(20, k_step)], dim=1)
            x_in = Variable(x_in, requires_grad=False)
            model.init_center_c(x_in)
        model.init_center_c(None)

def save_model(model, fn_model):
    """Save Deep SVDD model to export_model."""
    net_dict = model.state_dict()
    torch.save({'c': model.c,
                'R': model.R,
                'net_dict': net_dict}, fn_model)

def load_model(model, fn_model):
    """Load Deep SVDD model from model_path."""
    model_dict = torch.load(fn_model)
    model.c = model_dict['c']
    model.R = model_dict['R']
    model.load_state_dict(model_dict['net_dict'])

def get_radius(model, dataset, nu):
    model.eval()  # prep model for training
    dist = []
    with torch.no_grad():
        for n_batch, (input_data, targets) in enumerate(dataset):
            data = Variable(input_data).to(device)
            y = Variable(targets).to(device)
            model0(data, y)
            x_in = model0.z_normal
            x_in = torch.cat(x_in, dim=2)
            x_in = torch.cat([x_in[:, :, i, :] for i in range(20, k_step)], dim=1)
            x_in = Variable(x_in, requires_grad=False)
            model(x_in)
            dist.append(torch.norm(model.z-model.c, p=2, dim=2))
        dist = (torch.cat(dist, dim=1)).view(-1)
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

def model_train(model, dataset, optimizer, fn, maxepoch=10000):
    best_loss = 1e10
    last_step = 0
    for epoch in range(1, maxepoch + 1):
        print("train epoch {}".format(epoch))
        train_losses = []
        valid_losses = []
        model.train()  # prep model for training
        for n_batch, (input_data, targets) in enumerate(dataset):
            sys.stdout.write('\rEpoch: [%s--%s] ' % (epoch, n_batch))
            sys.stdout.flush()
            data = Variable(input_data).to(device)
            y = Variable(targets).to(device)
            with torch.no_grad():
                model0(data, y)

            optimizer.zero_grad()
            x_in = model0.z_normal
            x_in = torch.cat(x_in, dim=2)
            x_in = torch.cat([x_in[:, :, i, :] for i in range(20, k_step)], dim=1)
            x_in = Variable(x_in, requires_grad=False)
            model(x_in)
            loss = model.loss_criterion()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # grad norm clipping, only in pytorch version >= 1.10
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            # record training loss
            train_losses.append(loss.item())

        model.eval()  # prep model for evaluation
        with torch.no_grad():
            for n_batch, (input_data, targets) in enumerate(valid_loader):
                # forward pass: compute predicted outputs by passing inputs to the model
                data = Variable(input_data).to(device)
                y = torch.autograd.Variable(targets).to(device)
                model0(data, y)
                x_real = torch.cat([data[:, i, x_dim[0]:x_dim_sum] for i in range(20, k_step)], dim=0)
                optimizer.zero_grad()
                x_in = model0.z_normal
                x_in = torch.cat(x_in, dim=2)
                x_in = torch.cat([x_in[:, :, i, :] for i in range(20, k_step)], dim=1)
                x_in = Variable(x_in, requires_grad=False)
                model(x_in)
                loss = model.loss_criterion()
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        if train_loss<best_loss:
            best_loss=train_loss
            model.R = get_radius(model, dataset, nu)
            save_model(model, fn)
            last_step = 0
        else:
            last_step +=1

        print_msg = ("train epoch:{}   ".format(epoch) + "train_loss: {:.5f} ".format(train_loss)
                     + "valid_loss: {:.5f}".format(valid_loss) + "  best_loss: {:.5f}".format(best_loss)
                     + "  R: {:.5f}".format(model.R) +  "   last_step: {}".format(last_step))
        print(print_msg)

        if last_step>50 or epoch>=maxepoch:
            load_model(model, fn)
            return

train_loader, valid_loader, test_loader = dataset_loader(file_name, k_step, train_batch)

fn = './saves/SVDD_state_dict.pth'
if model_flag == 1:
    model = SVDD(x_dim, z_dim, h_dim*3, global_z_dim).to(device)
    load_model(model, fn)
else:
    model_per = Autoencoder(x_dim, z_dim, h_dim*3, global_z_dim).to(device)
    if 0:
        optimizer0 = torch.optim.Adam(model_per.parameters(), lr=learning_rate)
        model_pretrain(model_per, train_loader, optimizer0)
    else:
        perfn = "./saves/pertrain_autoencoder_dict.pth"
        model_per.load_state_dict(torch.load(perfn))
    model = SVDD(x_dim, z_dim, h_dim*3, global_z_dim).to(device)
    init_network_weights_from_pretraining(model, model_per)
    init_center_c(model,train_loader)
    save_model(model,fn)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model_train(model, train_loader, optimizer, fn)




