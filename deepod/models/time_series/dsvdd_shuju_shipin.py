# -*- coding: utf-8 -*-
"""
One-class classification
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import get_network
from torch.utils.data import DataLoader
import torch
import time
import numpy as np


class DeepSVDDTS(BaseDeepAD):
    """
    Deep One-class Classification for Anomaly Detection (ICML'18)
     :cite:`ruff2018deepsvdd`

    Parameters
    ----------
    epochs: int, optional (default=100)
        Number of training epochs

    batch_size: int, optional (default=64)
        Number of samples in a mini-batch

    lr: float, optional (default=1e-3)
        Learning rate

    network: str, optional (default='MLP')
        network structure for different data structures

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data
        deprecated when handling tabular data (network=='MLP')

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences
        deprecated when handling tabular data (network=='MLP')

    rep_dim: int, optional (default=128)
        Dimensionality of the representation space

    hidden_dims: list, str or int, optional (default='100,50')
        Number of neural units in hidden layers
            - If list, each item is a layer
            - If str, neural units of hidden layers are split by comma
            - If int, number of neural units of single hidden layer

    act: str, optional (default='ReLU')
        activation layer name
        choice = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']

    bias: bool, optional (default=False)
        Additive bias in linear layer

    epoch_steps: int, optional (default=-1)
        Maximum steps in an epoch
            - If -1, all the batches will be processed

    prt_steps: int, optional (default=10)
        Number of epoch intervals per printing

    device: str, optional (default='cuda')
        torch device,

    verbose: int, optional (default=1)
        Verbosity mode

    random_state： int, optional (default=42)
        the seed used by the random

    """

    def __init__(self, epochs=20, batch_size=256, lr=1e-5,
                 network='Transformer', seq_len=30, stride=10,
                 rep_dim=64, hidden_dims='512', act='GELU', bias=False,
                 n_heads=8, d_model=512, attn='cc_attn', pos_encoding='fixed', norm='LayerNorm',
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DeepSVDDTS, self).__init__(
            model_name='DeepSVDD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            network=network, seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        # parameters for Transformer
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = attn
        self.pos_encoding = pos_encoding
        self.norm = norm

        self.c = None
        return

    def training_prepare(self, X, y):
        time1 = time.time()
        # 时域频域数据混合
        # 将时域数据与频域数据按行拼接

        # 假设 X 是一个三维numpy数组，形状为 (batch_size, seq_len, features)
        # X = ...

        # 1. 对每个序列应用FFT，axis=1 表示对 seq_len 维度进行变换
        # FFT结果将是复数，但我们只取绝对值来获取频域的幅度信息
        X_freq = np.fft.fft(X, axis=1)

        # 2. 取频域数据的幅度和相位
        X_freq_amp = np.abs(X_freq)
        X_freq_phase = np.angle(X_freq)
        features = X.shape[-1]

        # 初始化交错结果的数组
        result = np.empty_like(X, shape=(X.shape[0], X.shape[1], X.shape[-1] * 3))

        # 交错排列
        for i in range(features):
            # 使用取模来循环选择张量

            result[..., 3 * i + 0] = X[..., i]
            result[..., 3 * i + 1] = X_freq_amp[..., i]
            result[..., 3 * i + 2] = X_freq_phase[..., i]




        X = result

        self.n_features = X.shape[2]





        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        network_params = {

            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        if self.network == 'Transformer':
            network_params['n_heads'] = self.n_heads
            network_params['d_model'] = self.d_model
            network_params['pos_encoding'] = self.pos_encoding
            network_params['norm'] = self.norm
            network_params['attn'] = self.attn
            network_params['seq_len'] = self.seq_len
        elif self.network == 'ConvSeq':
            network_params['seq_len'] = self.seq_len

        network_class = get_network(self.network)
        net = network_class(**network_params).to(self.device)

        # self.c = torch.randn(net.n_emb).to(self.device)
        self.c = self._set_c(net, train_loader)
        criterion = DSVDDLoss(c=self.c)

        if self.verbose >= 2:
            print(net)
        print('train_prepared_time', time.time() - time1)
        return train_loader, net, criterion

    def inference_prepare(self, X):
        time3 = time.time()
        batch_size, seq_len, features = X.shape
        # X = np.random.rand(batch_size, seq_len, features)

        # 1. 对每个序列应用FFT，axis=1 表示对 seq_len 维度进行变换
        X_freq = np.fft.fft(X, axis=1)

        # 2. 取频域数据的幅度和相位
        X_freq_amp = np.abs(X_freq)
        X_freq_phase = np.angle(X_freq)
        features = X.shape[-1]

        # 初始化交错结果的数组
        result = np.empty_like(X, shape=(X.shape[0], X.shape[1], X.shape[-1] * 3))

        # 交错排列
        for i in range(features):
            # 使用取模来循环选择张量

            result[..., 3 * i + 0] = X[..., i]
            result[..., 3 * i + 1] = X_freq_amp[..., i]
            result[..., 3 * i + 2] = X_freq_phase[..., i]




        X = result

        self.n_features = X.shape[2]

        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'none'

        print('inference_prepared_time', time.time() - time3)

        return test_loader

    def training_forward(self, batch_x, net, criterion):
        time1 = time.time()
        batch_x = batch_x.float().to(self.device)
        x_mean = torch.mean(batch_x, dim=1, keepdim=True)
        batch_x = batch_x- x_mean
        x_var = torch.var(batch_x, dim=1, keepdim=True) + 1e-5
        # print(x_var)
        batch_x = batch_x / torch.sqrt(x_var)
        print('training_forward_time1', time.time() - time1)
        time2 = time.time()
        z = net(batch_x)

        print('training_forward_time2', time.time() - time2)
        time3 = time.time()
        loss = criterion(z)
        print('training_forward_time3', time.time() - time3)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        time4 = time.time()
        batch_x = batch_x.float().to(self.device)
        x_mean = torch.mean(batch_x, dim=1, keepdim=True)
        batch_x = batch_x - x_mean
        x_var = torch.var(batch_x, dim=1, keepdim=True) + 1e-5
        # print(x_var)
        batch_x = batch_x / torch.sqrt(x_var)

        batch_z = net(batch_x)
        s = criterion(batch_z)
        print('inference_forward_time', time.time() - time4)
        return batch_z, s

    def _set_c(self, net, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        net.eval()
        z_ = []
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                z = net(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


class DSVDDLoss(torch.nn.Module):
    """

    Parameters
    ----------
    c: torch.Tensor
        Center of the pre-defined hyper-sphere in the representation space

    reduction: str, optional (default='mean')
        choice = [``'none'`` | ``'mean'`` | ``'sum'``]
            - If ``'none'``: no reduction will be applied;
            - If ``'mean'``: the sum of the output will be divided by the number of
            elements in the output;
            - If ``'sum'``: the output will be summed

    """

    def __init__(self, c, reduction='mean'):
        super(DSVDDLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, reduction=None):
        loss = torch.sum((rep - self.c) ** 2, dim=1)

        if reduction is None:
            reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss
