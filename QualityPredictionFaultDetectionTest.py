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
import os
from VAE_Model3 import VAEModel
from SVDD import SVDD, Autoencoder
from scipy.stats.distributions import chi2
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

x_dim = np.array([5, 6, 8, 8, 8, 8, 8, 7], dtype=np.int16)
x_dim_sum = np.sum(x_dim)
y_dim = 2
z_dim = 7
c_dim = 5
h_dim = 30
global_z_dim = 10
train_batch = 64
test_batch = 1000
M_sample =30
k_step = 50
n_epochs = 10000
learning_rate = 1e-4
clip = 10
fault_th =0.99
#fault_d = 2.33 #99%
fault_d = 1.65 #95%

def dataset_loader(filename, k_step, train_batch):
    file_data = np.array(pd.read_excel(filename, header=1))
    file_data = file_data[2000:7000, :]
    batch_random = np.random.permutation(len(file_data)-k_step+1)
    mean_std = torch.load('./saves/mean_std.t7')
    data_mean = mean_std['mean']
    data_std = mean_std['std']

    file_data = (file_data - data_mean)/data_std
    file_data[:, 0:x_dim[0]] = 1


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
    return test_loader, torch.from_numpy(data_mean).to(dtype=torch.float), torch.from_numpy(data_std).to(dtype=torch.float)

def load_model(model, fn_model):
    """Load Deep SVDD model from model_path."""
    model_dict = torch.load(fn_model)
    model.c = model_dict['c']
    model.R = model_dict['R']
    model.R = 0.350134
    model.load_state_dict(model_dict['net_dict'])


np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
device = torch.device("cuda:0")

fn0 = './saves/LSTM_state_dict_3.pth'
state_dict = torch.load(fn0)
model0 = VAEModel(x_dim, y_dim, h_dim, z_dim, c_dim, M_sample).to(device)
model0.load_state_dict(state_dict)

fn = './saves/SVDD_state_dict.pth'
model = SVDD(x_dim, z_dim, h_dim*3, global_z_dim).to(device)
load_model(model, fn)

file_name = '../data/data_01.xls'
test_loader, data_mean, data_std = dataset_loader(file_name, k_step, train_batch)
with torch.no_grad():
    model0.init_hstate()
    z_input = []
    z_normal = []
    mu_y = []
    std_y = []
    y_real = []
    for n_batch, (input_data, targets) in enumerate(test_loader):
        test_batch = input_data.size(0)
        input_data = Variable(input_data.squeeze())
        y_real.append(targets)
        for i in range(test_batch):
            y_mu, y_std = model0.seq_out(input_data[i, :].to(device))
            z_in = [(model0.z[i]-model0.z_pior_mu[i])/torch.exp(model0.z_pior_log_std[i]/2) for i in range(7)]
            z_in = Variable(torch.cat(z_in, dim=2), requires_grad=False)
            z_input.append(z_in)
            zt_normal = [((model0.z_mu[i] - model0.z_pior_mu[i])/ torch.exp(model0.z_pior_log_std[i]/2)).unsqueeze(0) for i in range(7)]
            zt_normal = torch.cat(zt_normal, dim=0)
            z_normal.append(zt_normal)
            mu_y.append(y_mu.unsqueeze(0))
            std_y.append(y_std.unsqueeze(0))
    z_normal = torch.cat(z_normal, dim=1)
    z_mean = torch.mean(z_normal[:,30:-1,:], dim=1)
    z_cov = [torch.zeros([z_normal.shape[2],z_normal.shape[2]]).to(device) for j in range(7)]
    for i in range(z_normal.shape[1]-30):
        z_cov = [z_cov[j] + ((z_normal[j,i+30,:]-z_mean[j,:]).unsqueeze(1)).mm((z_normal[j,i+30,:]-z_mean[j,:]).unsqueeze(0)) for j in range(7)]
    z_cov = [(z_cov[j]/(z_normal.shape[1]-30-1)).unsqueeze(0) for j in range(7)]
    z_cov = torch.cat(z_cov, dim=0)

    y_mean = data_mean[-y_dim:]
    y_std = data_std[-y_dim:]
    y_real = torch.cat(y_real, dim=0)* y_std + y_mean
    mu_y = torch.cat(mu_y,dim=0)* y_std + y_mean
    std_y = torch.cat(std_y, dim=0) * y_std
    data_df = pd.DataFrame((torch.cat([y_real, mu_y,std_y], dim=1)).data.cpu().numpy())
    data_df.columns = [('y' + str(x)) for x in range(y_dim)] + [('mu' + str(x)) for x in range(y_dim)] + [('std' + str(x)) for x in range(y_dim)]
    writer = pd.ExcelWriter('y_pre.xlsx')
    data_df.to_excel(writer, 'page_1')
    writer.save()

#file_name = '../data/fault_gap.xls'
file_name = '../data/fault_cooling.xls'
test_loader, _, _ = dataset_loader(file_name, k_step, train_batch)
with torch.no_grad():
    model0.init_hstate()
    z_input = []
    z_normal = []
    dist = []
    SPE = []
    T2 = []
    for n_batch, (input_data, targets) in enumerate(test_loader):
        test_batch = input_data.size(0)
        input_data = Variable(input_data.squeeze())
        for i in range(test_batch):
            model0.seq_out(input_data[i, :].to(device))
            z_in = [(model0.z[i]-model0.z_pior_mu[i])/torch.exp(model0.z_pior_log_std[i]/2) for i in range(7)]
            z_in = Variable(torch.cat(z_in, dim=2), requires_grad=False)
            z_input.append(z_in)
            zt_normal = [(model0.z_mu[i] - model0.z_pior_mu[i])/ torch.exp(model0.z_pior_log_std[i]/2) for i in range(7)]
            T2_t = [torch.sum((model0.z_mu[i]-model0.z_pior_mu[i])**2/torch.exp(model0.z_pior_log_std[i]),dim=1) for i in range(7)]
            SPE_t = [torch.mean(torch.sum((model0.x[i] - model0.x_mu[i])**2/ torch.exp(model0.x_log_std[i]),dim=2),dim=0) for i in range(7)]
            zt_normal = (torch.cat(zt_normal, dim=0)).unsqueeze(1)
            T2_t = (torch.cat(T2_t, dim=0)).unsqueeze(0)
            SPE_t = (torch.cat(SPE_t, dim=0)).unsqueeze(0)
            T2.append(T2_t)
            SPE.append(SPE_t)
            z_normal.append(zt_normal)
        z_input=torch.cat(z_input, dim=1)
        model(z_input)
        dist.append(torch.norm(model.z-model.c, p=2, dim=2))

    z_normal=torch.cat(z_normal, dim=1)
    #T2 = (torch.sum(z_normal**2, dim=2)).t()
    T2 = [(torch.sum(((z_normal[i,:,:]-z_mean[i,:]).mm(z_cov[i,:,:].inverse()))*(z_normal[i,:,:]-z_mean[i,:]), dim = 1)).unsqueeze(1) for i in range(7)]
    T2= torch.cat(T2,dim=1)
    SPE=torch.cat(SPE, dim=0)
    SPE_SUM = torch.sum(SPE, dim=1)
    dist = torch.cat(dist, dim=1)
    p_T2 = np.zeros([test_batch, 7])
    p_SPE = np.zeros([test_batch, 7])
    p_SPE_SUM = chi2.cdf(SPE_SUM.data.cpu().numpy(), np.sum(x_dim)-x_dim[0])
    for i in range(7):
        p_T2[:,i] = T2[:, i].data.cpu().numpy()
        p_SPE[:, i] = chi2.cdf(SPE[:, i].data.cpu().numpy(), x_dim[i+1])

    Dict = {}
    Dict['dist'] = (torch.mean(dist, dim=0).data.cpu().numpy()).tolist()
    Dict['R'] = ((np.ones(dist.shape[1]))*model.R).tolist()
    Dict['SPE_SUM'] = (p_SPE_SUM).tolist()
    Dict['T2_1'] = (p_T2[:, 0]).tolist()
    Dict['T2th_1'] = ((np.ones(dist.shape[1]))*chi2.ppf(fault_th, z_dim)).tolist()
    Dict['SPE_1'] = (p_SPE[:, 0]).tolist()
    Dict['SPEth_1'] = ((np.ones(dist.shape[1])) *fault_th).tolist()

    Dict['T2_2'] = (p_T2[:, 1]).tolist()
    Dict['T2th_2'] = ((np.ones(dist.shape[1])) * chi2.ppf(fault_th, z_dim)).tolist()
    Dict['SPE_2'] = (p_SPE[:, 1]).tolist()
    Dict['SPEth_2'] = ((np.ones(dist.shape[1])) * fault_th).tolist()

    Dict['T2_3'] = (p_T2[:, 2]).tolist()
    Dict['T2th_3'] = ((np.ones(dist.shape[1])) * chi2.ppf(fault_th, z_dim)).tolist()
    Dict['SPE_3'] = (p_SPE[:, 2]).tolist()
    Dict['SPEth_3'] = ((np.ones(dist.shape[1])) * fault_th).tolist()

    Dict['T2_4'] = (p_T2[:, 3]).tolist()
    Dict['T2th_4'] = ((np.ones(dist.shape[1])) * chi2.ppf(fault_th, z_dim)).tolist()
    Dict['SPE_4'] = (p_SPE[:, 3]).tolist()
    Dict['SPEth_4'] = ((np.ones(dist.shape[1])) * fault_th).tolist()

    Dict['T2_5'] = (p_T2[:, 4]).tolist()
    Dict['T2th_5'] = ((np.ones(dist.shape[1])) *chi2.ppf(fault_th, z_dim)).tolist()
    Dict['SPE_5'] = (p_SPE[:, 4]).tolist()
    Dict['SPEth_5'] = ((np.ones(dist.shape[1])) * fault_th).tolist()

    Dict['T2_6'] = (p_T2[:, 5]).tolist()
    Dict['T2th_6'] = ((np.ones(dist.shape[1])) *chi2.ppf(fault_th, z_dim)).tolist()
    Dict['SPE_6'] = (p_SPE[:, 5]).tolist()
    Dict['SPEth_6'] = ((np.ones(dist.shape[1])) * fault_th).tolist()

    Dict['T2_7'] = (p_T2[:, 6]).tolist()
    Dict['T2th_7'] = ((np.ones(dist.shape[1])) * chi2.ppf(fault_th, z_dim) ).tolist()
    Dict['SPE_7'] = (p_SPE[:, 6]).tolist()
    Dict['SPEth_7'] = ((np.ones(dist.shape[1])) * fault_th).tolist()
    df = pd.DataFrame(Dict)
    df.to_excel("Fault_detection_global.xls")

    with PdfPages('./img/global_local.pdf') as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(figsize=(10, 10))  # 图片大小为10*10
        plt.subplot(211)
        plt.plot(torch.mean(dist, dim=0).data.cpu().numpy(), color="blue", linewidth=0.5)
        plt.hlines(model.R, 0, dist.shape[1], colors="r", linestyles="dashed")
        plt.title('svdd dist')
        plt.subplot(212)
        plt.plot(p_SPE_SUM, color="blue", linewidth=0.5)
        plt.hlines(fault_th, 0, dist.shape[1], colors="r", linestyles="dashed")
        plt.title('svdd SPE')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        for i in range(7):
            fig = plt.figure()
            ax = fig.add_subplot(figsize=(10, 10))  # 图片大小为10*10
            plt.subplot(211)
            plt.plot(chi2.cdf(SPE[:, i].data.cpu().numpy(), x_dim[i + 1]))
            plt.hlines(fault_th, 0, SPE.shape[0], colors="r", linestyles="dashed")
            plt.title('SPE_%d' % (i + 1))
            plt.subplot(212)
            plt.plot(chi2.cdf(T2[:, i].data.cpu().numpy(), z_dim))
            plt.hlines(fault_th, 0, T2.shape[0], colors="r", linestyles="dashed")
            plt.title('T2_%d' % (i + 1))
            plt.tight_layout()
            pdf.savefig()
            plt.close()