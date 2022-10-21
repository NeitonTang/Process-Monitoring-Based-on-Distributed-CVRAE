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


x_dim = np.array([5, 6, 8, 8, 8, 8, 8, 7], dtype=np.int16)
x_dim_sum = np.sum(x_dim)
y_dim = 2
z_dim = 7
c_dim = 5
h_dim = 30
train_batch = 512
test_batch = 1000
k_step = 50
file_name = '../data/data_01.xls'
n_epochs = 10000
learning_rate = 5e-4
clip = 10

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


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

    #test_data = np.reshape(file_data[int(len(file_data)/2): (int(len(file_data)/2)+test_batch), 0:x_dim], (1, test_batch, x_dim))
    #test_target = np.reshape(file_data[int(len(file_data) / 2): (int(len(file_data) / 2) + test_batch), x_dim:x_dim+y_dim],
     #                       (1, test_batch, y_dim))
    test_batch = file_data.shape[0]
    test_data = np.reshape(file_data[:, 0:x_dim_sum], (1, test_batch, x_dim_sum))
    test_target = np.reshape(file_data[:, x_dim_sum:x_dim_sum+y_dim], (1, test_batch, y_dim))
    test_data = (torch.from_numpy(test_data)).to(dtype=torch.float)
    test_target = (torch.from_numpy(test_target)).to(dtype=torch.float)
    test_dataset = Data.TensorDataset(test_data, test_target)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch,
        shuffle=False)
    return train_loader, valid_loader, test_loader

torch.manual_seed(256)
np.random.seed(256)
train_loader, valid_loader, test_loader = dataset_loader(file_name, k_step, train_batch)
# Decide which device we want to run on
device = torch.device("cuda:0")
fn = './saves/LSTM_state_dict_3.pth'
#device = torch.device("cpu")
if 1:
    state_dict = torch.load(fn)
    model = VAEModel(x_dim, y_dim, h_dim, z_dim, c_dim).to(device)
    model.load_state_dict(state_dict)
else:
    model = VAEModel(x_dim, y_dim, h_dim, z_dim, c_dim).to(device)
    model.reset_parameters()

model.SetOptimizer()

#for name, param in model.named_parameters():
#    print(name)

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []

valid_result = []
avg_valid_result = []
plt.ion()
best_loss =1e10
for epoch in range(1, n_epochs + 1):
    print("epoch {}".format(epoch))
    ###################
    # train the model #
    ###################
    model.train()  # prep model for training
    for n_batch, (input_data, targets) in enumerate(train_loader):
        sys.stdout.write('\rEpoch: [%s--%s] ' % (epoch, n_batch))
        sys.stdout.flush()
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        data = Variable(input_data).to(device)
        y = Variable(targets).to(device)
        loss = model(data, y)
        #loss = model.loss_criterion()

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        # record training loss
        train_losses.append(loss.item())
        # print('====> Epoch: {}  itea {} Average loss: {:.4f}, loss: {:.4f},  {:.4f}'.format(
        #    epoch, n_batch, loss.item(), recon_loss.item(), kl_loss.item()))

    model.eval()  # prep model for evaluation
    with torch.no_grad():
        for n_batch, (input_data, targets) in enumerate(valid_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            data = Variable(input_data).to(device)
            y = torch.autograd.Variable(targets).to(device)
            loss = model(data, y)
            #loss = model.loss_criterion()
            # record validation loss
            valid_losses.append(loss.item())

            result = np.array([model.sub_vae1.x_error().item(), model.sub_vae2.x_error().item(), model.sub_vae3.x_error().item(),
                               model.sub_vae4.x_error().item(), model.sub_vae5.x_error().item(), model.sub_vae6.x_error().item(),
                               model.sub_vae7.x_error().item(), model.y_error().item(), 0,
                               model.sub_vae1.recon_loss.item(), model.sub_vae2.recon_loss.item(), model.sub_vae3.recon_loss.item(),
                               model.sub_vae4.recon_loss.item(), model.sub_vae5.recon_loss.item(), model.sub_vae6.recon_loss.item(),
                               model.sub_vae7.recon_loss.item(), model.recon_loss_y.item(), 0,
                               model.sub_vae1.kl_loss.item(), model.sub_vae2.kl_loss.item(), model.sub_vae3.kl_loss.item(),
                               model.sub_vae4.kl_loss.item(), model.sub_vae5.kl_loss.item(), model.sub_vae6.kl_loss.item(),
                               model.sub_vae7.kl_loss.item(), 0])
            valid_result.append(result)


    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    valid_results = np.average(valid_result, axis=0)
    avg_valid_result.append(valid_results)

    epoch_len = len(str(n_epochs))

    print_msg = ("z dim:{}   ".format(model.h_dim) + "epoch:{}   ".format(epoch)
                 +"train_loss: {:.5f} ".format(train_loss)
                 +"valid_loss: {:.5f}".format(valid_loss))

    print(print_msg)
    print(valid_results)

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []
    valid_result = []

    #fn = './saves/LSTM_state_dict_3.pth'
    if train_loss<best_loss:
        best_loss=train_loss
        torch.save(model.state_dict(), fn)

    plt.clf()
    plt.plot(np.array(avg_train_losses), color="blue")
    plt.plot(np.array(avg_valid_losses), color="red")
    plt.ylim(-50, 0)
    plt.show()
    plt.pause(0.1)
    '''
    plt.ion()
    if epoch %20 == 1:
        with torch.no_grad():
            for n_batch, (input_data, targets) in enumerate(test_loader):
                test_batch = input_data.size(1)
                input_data = Variable(input_data.squeeze())
                y_pre = torch.zeros([test_batch,y_dim], dtype=torch.float)
                y_pre_std = torch.zeros([test_batch, y_dim], dtype=torch.float)
                model.init_hstate()
                for i in range(test_batch):
                    y_pre[i, :], y_pre_std[i, :] = model.seq_out(input_data[i, :].to(device))

                plt.clf()
                y = targets.squeeze().data.cpu().numpy()
                y_p = y_pre.data.cpu().numpy()
                y_std = y_pre_std.data.cpu().numpy()
                x = np.array(range(y.shape[0]))
                plt.subplot(211)
                plt.plot(y[:, 0].reshape(-1, 1), color="blue")
                plt.plot(y_p[:, 0].reshape(-1, 1), color="red", alpha=0.5)
                plt.fill_between(x, (y_p[:, 0] - 2.33 * y_std[:, 0]), (y_p[:, 0] + 2.33 * y_std[:, 0]), color="orange",
                                 alpha=0.5)
                plt.title('1')
                plt.subplot(212)
                plt.plot(y[:, 1].reshape(-1, 1), color="blue")
                plt.plot(y_p[:, 1].reshape(-1, 1), color="red", alpha=0.5)
                plt.fill_between(x, (y_p[:, 1] - 2.33 * y_std[:, 1]), (y_p[:, 1] + 2.33 * y_std[:, 1]), color="orange",
                                 alpha=0.5)
                plt.title('2')
                plt.show()
                plt.pause(0.1)
        '''

