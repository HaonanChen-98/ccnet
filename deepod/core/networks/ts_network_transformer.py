import torch
import math
import time
from typing import Optional, Tuple
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from deepod.core.networks.network_utility import _instantiate_class, _handle_n_hidden


# def INF(B, H, W):
#     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


# class CrissCrossAttention(torch.nn.Module):
#     """ Criss-Cross Attention Module"""
#     def __init__(self, in_dim, kernel_size=3):
#         super(CrissCrossAttention,self).__init__()
#
#         self.query_linear = TokenEmbedding(n_features=in_dim, d_model=in_dim,
#                                            kernel_size=3, bias=False)
#         self.key_linear = TokenEmbedding(n_features=in_dim, d_model=in_dim,
#                                            kernel_size=3, bias=False)
#         self.value_linear = TokenEmbedding(n_features=in_dim, d_model=in_dim,
#                                            kernel_size=3, bias=False)
#
#         self.softmax = torch.nn.Softmax(dim=3)
#         self.INF = INF
#         self.gamma = torch.nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         x = x.permute(1, 0, 2)  # (batchsize, seq_len, dim) = (32, 30, 64)
#         m_batchsize, height, width = x.size()
#
#         proj_query = self.query_linear(x).unsqueeze(1)
#         proj_key = self.key_linear(x).unsqueeze(1)
#         proj_value = self.value_linear(x).unsqueeze(1)
#
#         # all columns: (batchsize*weight, height, channels) = (2048=32*64, 64, 1)
#         proj_query_h = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,  -1, height).permute(0, 2, 1)
#         # all rows: (batchsize*height, weight, channels) = (960=32*30, 30, 1)
#         proj_query_w = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height, -1, width).permute(0, 2, 1)
#
#         proj_key_h = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1, height)
#         proj_key_w = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1, width)
#
#         proj_value_h = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
#         proj_value_w = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
#
#
#         # (batchsize, weight, height, height) = (32, 64, 30, 30)
#         energy_h = torch.bmm(proj_query_h, proj_key_h).view(m_batchsize,width,height,height)
#         # (batchsize, height, weight, height) = (32, 30, 64, 30)
#         # each position is a 30-dimensional vector, representing their correlations with other columns
#         energy_h = energy_h.permute(0,2,1,3)
#         # (batchsize, height, weight, weight) = (32, 30, 64, 64)
#         # each position is a
#         energy_w = torch.bmm(proj_query_w, proj_key_w).view(m_batchsize,height,width,width)
#
#         # attention, (batch_size, height, width, height+width) = (32, 30, 64, 30+64)
#         concate = self.softmax(torch.cat([energy_h, energy_w], 3))
#
#         # attach attention to value
#         att_h = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width, height, height)
#         att_w = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height, width, width)
#         out_h = torch.bmm(proj_value_h, att_h.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
#         out_w = torch.bmm(proj_value_w, att_w.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
#
#         output = self.gamma * (out_h + out_w)
#         output = output.squeeze(1)
#         # output = output + x # residual connection is implemented in EncoderLayer
#
#         output = output.permute(1, 0, 2)
#
#         return output


class CrissCrossAttention(torch.nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim, cut_freq=3, DSR=4, kernel_size=3):
        super(CrissCrossAttention, self).__init__()

        self.query_linear = torch.nn.Linear(in_dim, in_dim, bias=False)
        self.key_linear = torch.nn.Linear(in_dim, in_dim, bias=False)
        self.value_linear = torch.nn.Linear(in_dim, in_dim, bias=False)

        # self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = torch.nn.Softmax(dim=3)
        # self.INF = INF
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        #频域参数
        self.DSR = DSR
        self.dominance_freq = cut_freq  # 720/24
        self.length_ratio = DSR
        self.freq_upsampler = torch.nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
                torch.cfloat)  # complex layer for frequency upcampling]

    def forward(self, x):

        # t1 = time.time()
        # print('------------------')
        # print(x.shape) # (seq_len, batch_size, dimension)
        features = int(x.shape[-1] / 2)
        x = x[:, :, 0:features]

        x_fre = x.permute(1, 0, 2)
        x_fre = x_fre.float()[:, ::self.DSR, :]


        x_mean = torch.mean(x, dim=0, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=0, keepdim=True) + 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        x_mean_fre = torch.mean(x_fre, dim=1, keepdim=True)
        x_fre = x_fre - x_mean_fre
        x_var_fre = torch.var(x_fre, dim=1, keepdim=True) + 1e-5
        # print(x_var)
        x_fre = x_fre / torch.sqrt(x_var_fre)



        low_specx = torch.fft.rfft(x_fre, dim=1)

        low_specx[:, self.dominance_freq:] = 0


        # low_x=torch.fft.irfft(low_specx, dim=1)
        low_specx = low_specx[:, 0:self.dominance_freq, :]
        # print(low_specx.dtype)
        # print(low_specx.permute(0,2,1))

            # print(low_specx.permute(0,2,1).size())
        low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        #'seq_len': self.win_size//self.DSR, ,
        # 'cut_freq':self.cutfreq,'pred_len':self.win_size-self.win_size//self.DSR}
        low_specxy = torch.zeros(
            [low_specxy_.size(0), int(x.shape[0] //2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_

        low_xy = torch.fft.irfft(low_specxy, dim=1)

        low_xy = low_xy * self.length_ratio  # compemsate the length change
        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy = (low_xy) * torch.sqrt(x_var_fre) + x_mean_fre
        xy = xy.permute(1, 0, 2)
        result = torch.empty(size=(x.shape[0], x.shape[1], x.shape[-1] * 2), dtype=x.dtype,
                             device=x.device)
        # 交错排列
        for i in range(features):
            # 使用取模来循环选择张量

            result[..., 2 * i + 0] = x[..., i]
            result[..., 2 * i + 1] = xy[..., i]
        #

        x = result

        x = x.permute(1, 0, 2)  # (32, 30, 64)
        x = x.unsqueeze(1)  # batch_size, channels, height, weight =  (32, 1, 30, 64)
        m_batchsize, _, height, width = x.size()

        proj_query = self.query_linear(x)
        # all columns: (batchsize*weight, height, channels) = (2048=32*64, 64, 1)
        proj_query_h = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        # all rows: (batchsize*height, weight, channels) = (960=32*30, 30, 1)
        proj_query_w = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)

        proj_key = self.key_linear(x)
        proj_key_h = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_w = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_linear(x)
        proj_value_h = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_w = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)

        # (batchsize, weight, height, height) = (32, 64, 30, 30)
        energy_h = torch.bmm(proj_query_h, proj_key_h).view(m_batchsize, width, height, height)
        # (batchsize, height, weight, height) = (32, 30, 64, 30)
        # each position is a 30-dimensional vector, representing their correlations with other columns
        energy_h = energy_h.permute(0, 2, 1, 3)
        # (batchsize, height, weight, weight) = (32, 30, 64, 64)
        # each position is a
        energy_w = torch.bmm(proj_query_w, proj_key_w).view(m_batchsize, height, width, width)

        # attention, (batch_size, height, width, height+width) = (32, 30, 64, 30+64)
        concate = self.softmax(torch.cat([energy_h, energy_w], 3))

        # print(energy_H.shape, energy_W.shape)
        # print(concate.shape)
        # t3 = time.time()
        # print(t3 - t2)

        # attach attention to value
        att_h = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_w = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_h = torch.bmm(proj_value_h, att_h.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_w = torch.bmm(proj_value_w, att_w.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        # print(att_H.shape) # (batch*width, height, height)
        # print(att_W.shape)
        # print(out_H.size(),out_W.size())

        # output = self.gamma * (out_H + out_W) + x
        output = self.gamma * (out_h + out_w)

        output = output.squeeze(1)
        output = output.permute(1, 0, 2)


        return output


class TokenEmbedding(torch.nn.Module):
    def __init__(self, n_features, d_model, kernel_size=3, bias=False):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = torch.nn.Conv1d(in_channels=n_features, out_channels=d_model,
                                         kernel_size=(kernel_size,), padding=padding,
                                         padding_mode='circular', bias=bias)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedPositionalEncoding(torch.nn.Module):
    r"""
        Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
        adapted from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)

    Args:
        d_model:
            the embed dim (required).

        dropout:
            the dropout value (default=0.1).

        max_len:
            the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        # self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).float()  # positional encoding
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Parameters
        ----------
        x: torch.Tensor, required
            shape= (sequence length, batch size, embed dim)
            the sequence fed to the positional encoder model (required).

        Returns
        -------
        output: torch.Tensor, required
            shape=(sequence length, batch size, embed dim)
        """
        x = self.pe[:, :x.size(1)]
        return x


class LearnablePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = torch.nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        torch.nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(torch.nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='ReLU',
                 layer_norm_eps=1e-5, attn='self_attn', batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()

        self.attn = attn
        if attn == 'self_attn':
            self.attn_model = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                          batch_first=batch_first,
                                                          **factory_kwargs)

        elif attn == 'cc_attn':
            self.attn_model = CrissCrossAttention(d_model)
        else:
            raise NotImplementedError('')

        # Implementation of Feedforward model
        # self.conv1 = torch.nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1)
        # self.conv2 = torch.nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1)

        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        assert activation in ['ReLU', 'GELU'], \
            f"activation should be ReLU/GELU, not {activation}"
        self.activation = _instantiate_class("torch.nn.modules.activation", activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:

        if self.attn == 'self_attn':
            x = self.attn_model(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]

        elif self.attn == 'cc_attn':
            recurrence = 2
            for kk in range(recurrence):
                x = self.attn_model(x)

        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerBatchNormEncoderLayer(torch.nn.modules.Module):
    r"""
    This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multi-head attention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attn='self_attn', activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()

        self.attn = attn
        if attn == 'self_attn':
            self.attn_model = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        elif attn == 'cc_attn':
            self.attn_model = CrissCrossAttention(d_model)
        else:
            raise NotImplementedError('')

        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.BatchNorm1d(d_model,
                                          eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = torch.nn.BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        assert activation in ['ReLU', 'GELU'], \
            f"activation should be ReLU/GELU, not {activation}"
        self.activation = _instantiate_class("torch.nn.modules.activation", activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.attn == 'self_attn':
            src2 = self.attn_model(src, src, src,
                                   attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]

        elif self.attn == 'cc_attn':
            recurrence = 2
            for _ in range(recurrence):
                src2 = self.attn_model(src)
        else:
            raise NotImplementedError('')

        # src2 = self.self_attn(src, src, src,
        #                       attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)

        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(torch.nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, n_features, n_output=20, seq_len=100, d_model=128,
                 n_heads=8, n_hidden='128', dropout=0.1, device='cuda',
                 attn='self_attn', token_encoding='convolutional', pos_encoding='fixed',
                 activation='GELU', bias=False,
                 norm='LayerNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()
        self.device = device
        self.max_len = seq_len
        self.d_model = d_model
        n_hidden, n_layers = _handle_n_hidden(n_hidden)

        # parameter check
        assert token_encoding in ['linear', 'convolutional'], \
            f"use 'linear' or 'convolutional', {token_encoding} is not supported in token_encoding"
        assert pos_encoding in ['learnable', 'fixed'], \
            f"use 'learnable' or 'fixed', {pos_encoding} is not supported in pos_encoding"
        assert norm in ['LayerNorm', 'BatchNorm'], \
            f"use 'learnable' or 'fixed', {norm} is not supported in norm"

        if token_encoding == 'linear':
            self.project_inp = torch.nn.Linear(n_features, d_model, bias=bias)
        elif token_encoding == 'convolutional':
            self.project_inp = TokenEmbedding(n_features, d_model, kernel_size=3, bias=bias)

        if pos_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(d_model, dropout=dropout * (1.0 - freeze), max_len=seq_len)
        elif pos_encoding == "fixed":
            self.pos_enc = FixedPositionalEncoding(d_model, dropout=dropout * (1.0 - freeze), max_len=seq_len)

        encoder_layer = None
        if norm == 'LayerNorm':
            # d_model -> n_hidden -> d_model
            encoder_layer = TransformerEncoderLayer(d_model, n_heads,
                                                    n_hidden, dropout * (1.0 - freeze),
                                                    attn=attn,
                                                    activation=activation)
        elif norm == 'BatchNorm':
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, n_heads,
                                                             n_hidden, dropout * (1.0 - freeze),
                                                             attn=attn,
                                                             activation=activation)

        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        assert activation in ['ReLU', 'GELU'], \
            f"activation should be ReLU/GELU, not {activation}"
        self.act = _instantiate_class("torch.nn.modules.activation", activation)

        self.dropout = torch.nn.Dropout(dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        # self.output_layer = torch.nn.Linear(d_model * seq_len, n_output, bias=bias)
        self.output_layer = torch.nn.Linear(d_model, n_output, bias=bias)
        self.padding_masks = (torch.ones(64,30, dtype=torch.uint8)).to(self.device)


    def forward(self, X, padding_masks=None):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]

        # inp = X.permute(1, 0, 2)
        # inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        # inp = self.pos_enc(inp)  # add positional encoding

        # means = X.mean(1, keepdim=True).detach()
        # stdev = torch.sqrt(torch.var(X, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # X = X - means
        # X /= stdev

        # data embedding
        inp = self.project_inp(X) + self.pos_enc(X)
        # inp = self.dropout(inp)
        inp = inp.permute(1, 0, 2)

        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        time1 = time.time()
        output = self.transformer_encoder(inp,
                                          src_key_padding_mask=~padding_masks if padding_masks is not None else None)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        print('ccnetshijian', time.time() - time1)
        if self.padding_masks.shape[0] != output.shape[0]:
            self.padding_masks = (torch.ones(X.shape[0], X.shape[1], dtype=torch.uint8))
            self.padding_masks = self.padding_masks.to(X.device)
            print(X.shape[0], X.shape[1], X.device)
            print('masknetshijian+++++++++++++++++++++++++++++++', time.time() - time1)

        # if padding_masks is None:
        #     padding_masks = (torch.ones(X.shape[0], X.shape[1], dtype=torch.uint8))
        #     print('masknetshijian1', time.time() - time1, padding_masks )
        #     padding_masks = padding_masks.to(X.device)
        #     print(X.shape[0], X.shape[1], X.device)
        print('masknetshijian2', time.time() - time1)

        # Output
        output = output * self.padding_masks.unsqueeze(-1)  # (batch_size, seq_len, 1) zero-out padding embeddings
        output = output[:, -1]  # (batch_size, d_model)
        # output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        print('layernetshijian', time.time() - time1)
        return output


if __name__ == '__main__':
    import time

    a = torch.randn(32, 100, 19)

    # t1 = time.time()
    # model = TSTransformerEncoder(n_features=19, seq_len=100,
    #                              token_encoding='convolutional', attn='self_attn',
    #                              d_model=64, n_heads=8, n_hidden='512',
    #                              n_output=128)
    # b = model(a)
    # print(b.shape)
    # print(time.time() - t1)

    model2 = TSTransformerEncoder(n_features=19, seq_len=100,
                                  token_encoding='convolutional', attn='cc_attn',
                                  d_model=64, n_heads=8, n_hidden='512',
                                  n_output=128)

    t1 = time.time()
    b = model2(a)
    print(b.shape)

    print(time.time() - t1)
