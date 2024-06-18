import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = nn.ModuleList([ 
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.enc_in)
        ]) if configs.channel_independence else nn.Linear(configs.seq_len, configs.pred_len)
        
        self.dropout = nn.Dropout(0.2)
        self.rev = RevIN(configs.enc_in) if configs.revin else None
        self.individual = configs.channel_independence
        self.task_name = configs.task_name

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # x: [B, L, D]
            x = self.rev(x, 'norm') if self.rev else x
            x = self.dropout(x)
            if self.individual:
                pred = torch.zeros_like(x_dec)
                for idx, proj in enumerate(self.Linear):
                    pred[:, :, idx] = proj(x[:, :, idx])
            else:
                pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
            pred = self.rev(pred, 'denorm') if self.rev else pred

            return pred, 0