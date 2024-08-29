import torch
from torch import nn
from layers.Embed import PatchEmbedding, DataEmbedding
from mamba_ssm import Mamba
from layers.st_moe import MoE, RMSNorm
import math


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class Experts(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, hidden_mult=4, mult_bias=True, prenorm=False):
        super().__init__()
        # other initializations
        self.mamba = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)

        layers = []
        if prenorm:
            layers.append(RMSNorm(dim))
        layers.extend([
            self.mamba,
        ])

        self.net = nn.Sequential(*layers)

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, Mamba):
            # Initialize weights and biases for layers inside Mamba
            for m in module.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                    dim = m.weight.shape[0]
                    std = dim ** -0.5
                    m.weight.data.uniform_(-std, std)
                    if m.bias is not None:
                        m.bias.data.uniform_(-std, std)

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, configs, d_state=16, d_conv=4, expand=3, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        self.use_norm = configs.use_norm

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.moe = MoE(
                        dim = configs.d_model,
                        num_experts = configs.num_experts,               # increase the experts (# parameters) of your model without increasing computation
                        gating_top_n = configs.num_gates,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
                        threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
                        threshold_eval = 0.2,
                        capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                        capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                        balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
                        router_z_loss_coef = 1e-3,      # loss weight for router z-loss
                        expert_hidden_mult=configs.moe_hidden_factor
                    )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.projection = nn.Linear(self.head_nf, configs.c_out, bias=True)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        if self.task_name == 'classification':
            self.act = nn.GELU()
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model  * configs.seq_len, configs.num_class)
            self.layer_norm = nn.LayerNorm(configs.d_model)
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        cka_in, n_vars = self.patch_embedding(x_enc)

        # MoE
        # z: [bs * nvars x patch_num x d_model]
        cka_out, total_aux_loss, balance_loss, router_z_loss = self.moe(cka_in)
        enc_out = cka_out
        # enc_out = self.mamba(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)       

        # Decoder
        dec_out = self.head(enc_out)  # z: [B, C, L]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # total_aux_loss = 0
        return dec_out, total_aux_loss
    
    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]

        # MoE
        enc_out, total_aux_loss, balance_loss, router_z_loss = self.moe(enc_out)
        enc_out = self.layer_norm(enc_out)
        enc_out = self.act(enc_out)  
        enc_out = enc_out * x_mark_enc.unsqueeze(-1)

        # Output
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output, total_aux_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, total_aux_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], total_aux_loss  # [B, L, D]
        if self.task_name == 'classification':
            dec_out, total_aux_loss = self.classification(x_enc, x_mark_enc)
            return dec_out, total_aux_loss  # [B, N]
        return None
