import torch
from torch import nn
from layers.Embed import PatchEmbedding
from mamba_ssm import Mamba
# from mixture_of_experts import MoE, HeirarchicalMoE
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
        # self.mlp = nn.Sequential[nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim)]

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
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


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

        # self.mamba = Mamba(configs.d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # MoE
        # self.expert = Experts(configs.d_model, d_state=d_state, d_conv=d_conv, expand=expand).to("cuda")
        # self.moe = MoE(
        #                 dim = configs.d_model,
        #                 num_experts = 4,               # increase the experts (# parameters) of your model without increasing computation
        #                 experts=self.expert,
        #                 hidden_dim = configs.d_model * 2,           # size of hidden dimension in each expert, defaults to 4 * dimension
        #                 activation = nn.GELU,      # use your preferred activation, will default to GELU
        #                 second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
        #                 second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
        #                 second_threshold_train = 0.2,
        #                 second_threshold_eval = 0.2,
        #                 capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
        #                 capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
        #                 loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
        #             )


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
                        # experts=self.expert,
                        expert_hidden_mult=configs.moe_hidden_factor
                    )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.projection = nn.Linear(self.head_nf, configs.c_out, bias=True)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        # elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
        #     self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
        #                             head_dropout=configs.dropout)
        # elif self.task_name == 'classification':
        #     self.flatten = nn.Flatten(start_dim=-2)
        #     self.dropout = nn.Dropout(configs.dropout)
        #     self.projection = nn.Linear(
        #         self.head_nf * configs.enc_in, configs.num_class)
        # Decoder
        if self.task_name == 'imputation':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
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
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # MoE
        # z: [bs * nvars x patch_num x d_model]
        enc_out, total_aux_loss, balance_loss, router_z_loss = self.moe(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)       

        # Decoder
        dec_out = self.head(enc_out)  # z: [B, C, L]
        dec_out = dec_out.permute(0, 2, 1)
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out, total_aux_loss
    
    
    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out, total_aux_loss, balance_loss, router_z_loss = self.moe(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
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
        if self.task_name == 'imputation':
            dec_out, total_aux_loss = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out, total_aux_loss  # [B, L, D]
        if self.task_name == 'classification':
            dec_out, total_aux_loss = self.classification(x_enc, x_mark_enc)
            return dec_out, total_aux_loss  # [B, N]
        return None
