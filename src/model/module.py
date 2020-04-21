import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__()
        self.model = nn.LSTM(*args, **kwargs)

    def forward(self, x):
        self.model.flatten_parameters()
        assert x.dim() == 3
        x = x.permute(0, 2, 1)
        o, _ = self.model(x)
        return o


class TCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channel, out_channels, dilation, padding, use_skip_connection):
        super().__init__()
        self.tcn_block = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channel, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channel, eps=1e-8),
            nn.Conv1d(hidden_channel, hidden_channel, 3, stride=1, groups=hidden_channel, padding=padding, dilation=dilation),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channel, eps=1e-8),
            nn.Conv1d(hidden_channel, out_channels, 1)
        )
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        """
            x: [channels, T]
        """
        if self.use_skip_connection:
            return x + self.tcn_block(x)
        else:
            return self.tcn_block(x)

