import torch
from torch import nn
import torch.nn.functional as F
from model_utils.tcn import TConv, MixProp
from model_utils.series_decompose import series_decomp


class MLPFeatureEncoder(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.win_size = config.win_size
        self.seq_encoder = MLPEncoder(config.seq_len - config.win_size, config.hidden_dim)

    def forward(self, x):
        context_x = x[:, :-self.win_size, :]
        suspect_x = x[:, self.win_size:, :]

        context_state = self.seq_encoder(context_x)
        suspect_state = self.seq_encoder(suspect_x)
        return context_state, suspect_state


class MLPEncoder(nn.Module):

    def __init__(self, seq_len, hidden_dim):
        super(MLPEncoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.individual = False

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.hidden_dim))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.hidden_dim))
        else:
            self.Linear_Seasonal = torch.nn.Sequential(nn.Linear(self.seq_len, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.Linear_Trend = torch.nn.Sequential(nn.Linear(self.seq_len, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, x):

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0) ,seasonal_init.size(1), self.pred_len]
                                          ,dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0) ,trend_init.size(1), self.pred_len]
                                       ,dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[: ,i ,:] = self.Linear_Seasonal[i](seasonal_init[: ,i ,:])
                trend_output[: ,i ,:] = self.Linear_Trend[i](trend_init[: ,i ,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x

class TCNFeatureEncoder_separate(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.win_size = config.win_size
        self.context_seq_encoder = SpatialTemporalEncoder_1D(config.feature_dim if config.target_dims is None else 1, config.seq_len-config.win_size, config.n_layers, config.hidden_dim,\
                                                  is_graph_conv=config.is_graph_conv, is_skip=config.is_skip, is_residual=config.is_residual, dilation_exp=2, dropout=config.dropout)

        self.suspect_seq_encoder = SpatialTemporalEncoder_1D(config.feature_dim if config.target_dims is None else 1, config.seq_len-config.win_size, config.n_layers, config.hidden_dim,\
                                                  is_graph_conv=config.is_graph_conv, is_skip=config.is_skip, is_residual=config.is_residual, dilation_exp=2, dropout=config.dropout)


    def forward(self, x):
        context_x = x[:, :-self.win_size, :]
        suspect_x = x[:, self.win_size:, :]

        context_state = self.context_seq_encoder(context_x)
        suspect_state = self.suspect_seq_encoder(suspect_x)
        return context_state, suspect_state


class TCNFeatureEncoder(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.win_size = config.win_size
        self.seq_encoder = SpatialTemporalEncoder_1D(config.feature_dim if config.target_dims is None else 1, config.seq_len-config.win_size, config.n_layers, config.hidden_dim,\
                                                  is_graph_conv=config.is_graph_conv, is_skip=config.is_skip, is_residual=config.is_residual, dilation_exp=2, dropout=config.dropout)
        # self.seq_encoder = SpatialTemporalEncoder(config.feature_dim if config.target_dims is None else 1, config.seq_len-config.win_size, config.n_layers, config.hidden_dim,\
        #                                           is_graph_conv=config.is_graph_conv, is_skip=config.is_skip, is_residual=config.is_residual, dilation_exp=2, dropout=config.dropout)
        #
        #self.graph_constructor = dynamic_graph_constructor(config.feature_dim, config.topK, config.hidden_dim, alpha=3)

        # self.graph_encoder = nn.Sequential(
        #     nn.Linear(1, config.hidden_dim), nn.LeakyReLU(), nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        # )

    def forward(self, x):
        context_x = x[:, :-self.win_size, :]
        suspect_x = x[:, self.win_size:, :]

        context_state = self.get_states(context_x)
        suspect_state = self.get_states(suspect_x)
        return context_state, suspect_state

    def get_states(self, x):
        # B, L, N = x.shape
        # emb = self.graph_encoder(x.transpose(1, 2).reshape(-1, L).unsqueeze(-1))[0][:, -1, :].reshape(B, N, -1)
        # adj = self.graph_constructor(emb)
        # x = self.seq_encoder(x, adj)
        x = self.seq_encoder(x)
        return x


class SpatialTemporalEncoder(nn.Module):

    def __init__(self, feature_dim, seq_length, n_layers, hidden_dim, is_skip=True, is_residual=True, is_graph_conv=True, dilation_exp=2, dropout=0.0):

        super().__init__()
        self.temporal_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.graph_convs = nn.ModuleList()

        self.n_layers = n_layers
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.is_skip = is_skip
        self.is_residual = is_residual
        self.is_graph_conv = is_graph_conv
        self.is_pad = True
        self.dropout = dropout
        kernel_sets = (2, 3, 5, 7)
        kernel_size = max(kernel_sets)
        self.start_conv = torch.nn.Conv2d(1, out_channels=hidden_dim, kernel_size=(1, 1))

        # if dilation_exp > 1:
        #     self.receptive_field = int(
        #         1 + (kernel_size - 1) * (dilation_exp ** self.n_layers - 1) / (dilation_exp - 1))
        # else:
        #     self.receptive_field = self.n_layers * (kernel_size - 1) + 1
        #
        # self.total_t_len = max(self.receptive_field, seq_length)

        def get_receptive_field():
            if dilation_exp > 1:
                self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exp ** self.n_layers - 1) / (dilation_exp - 1))
            else:
                self.receptive_field = self.n_layers * (kernel_size - 1) + 1

        self.total_t_len = self.seq_length

        get_receptive_field()
        while self.receptive_field > self.total_t_len:
            self.n_layers = self.n_layers - 1
            get_receptive_field()

        assert self.receptive_field <= self.total_t_len

        self.skip0 = torch.nn.Conv2d(hidden_dim, hidden_dim, \
                               kernel_size=(1, self.total_t_len), bias=True)
        self.skipE = torch.nn.Conv2d(hidden_dim, hidden_dim,
                               kernel_size=(1, self.total_t_len - self.receptive_field + 1), bias=True)

        new_dilation = 1
        for i in range(1, self.n_layers + 1):
            # dilated convolutions
            if dilation_exp > 1:
                rf_size_i = int(1 + (kernel_size - 1) * (dilation_exp ** i - 1) / (dilation_exp - 1))
            else:
                rf_size_i = 1 + i * (kernel_size - 1)
            t_len_i = self.total_t_len - rf_size_i + 1
            self.temporal_convs.append(TConv(hidden_dim, hidden_dim, kernel_set=kernel_sets, dilation_factor=new_dilation, dropout=dropout, conv_type="2D"))

            self.skip_convs.append(
                torch.nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, t_len_i)))

            self.bns.append(torch.nn.BatchNorm2d(hidden_dim))
            new_dilation *= dilation_exp

            self.graph_convs.append(MixProp(hidden_dim, hidden_dim, gdep=2, alpha=0.05))

    def forward(self, x, adj=None):

        x = x.transpose(1, 2).unsqueeze(1)

        if self.seq_length < self.receptive_field:
            if self.is_pad:
                x = F.pad(x, (self.receptive_field - self.seq_length, 0, 0, 0))

        x = self.start_conv(x)

        if self.is_skip:
           skip = self.skip0(F.dropout(x, self.dropout, training=self.training))

        for i in range(self.n_layers):
            if self.is_residual:
                residual = x
            x = self.temporal_convs[i](x, is_filter=False)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.is_skip:
                skip = skip + self.skip_convs[i](x)

            if self.is_graph_conv:
                x = self.graph_convs[i](x, adj)

            x = self.bns[i](x)
            x = torch.relu(x)
            if self.is_residual:
                x = x + residual[:, :, :, -x.size(3) :]

        if self.is_skip:
            skip =self.skipE(x)+skip
        else:
            skip = self.skipE(x)

        return skip.squeeze(-1).transpose(1, 2)


class SpatialTemporalEncoder_1D(nn.Module):

    def __init__(self, feature_dim, seq_length, n_layers, hidden_dim, is_skip=True, is_residual=True, is_graph_conv=True, dilation_exp=2, dropout=0.0):

        super().__init__()
        self.temporal_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.n_layers = n_layers
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.is_skip = is_skip
        self.is_residual = is_residual
        self.is_graph_conv = is_graph_conv
        self.is_pad = True
        self.dropout = dropout

        kernel_sets = (2, 3, 5, 7)
        kernel_size = max(kernel_sets)
        self.start_conv = torch.nn.Conv1d(feature_dim, out_channels=hidden_dim, kernel_size=1)

        # if dilation_exp > 1:
        #     self.receptive_field = int(
        #         1 + (kernel_size - 1) * (dilation_exp ** self.n_layers - 1) / (dilation_exp - 1))
        # else:
        #     self.receptive_field = self.n_layers * (kernel_size - 1) + 1
        #
        # self.total_t_len = max(self.receptive_field, seq_length)
        #
        def get_receptive_field():
            if dilation_exp > 1:
                self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exp ** self.n_layers - 1) / (dilation_exp - 1))
            else:
                self.receptive_field = self.n_layers * (kernel_size - 1) + 1

        self.total_t_len = self.seq_length

        get_receptive_field()
        while self.receptive_field > self.total_t_len:
            self.n_layers = self.n_layers - 1
            get_receptive_field()

        assert self.receptive_field <= self.total_t_len

        self.skip0 = torch.nn.Conv1d(hidden_dim, hidden_dim, \
                               kernel_size=self.total_t_len, bias=True)
        self.skipE = torch.nn.Conv1d(hidden_dim, hidden_dim,
                               kernel_size=self.total_t_len - self.receptive_field + 1, bias=True)

        new_dilation = 1
        for i in range(1, self.n_layers + 1):
            # dilated convolutions
            if dilation_exp > 1:
                rf_size_i = int(1 + (kernel_size - 1) * (dilation_exp ** i - 1) / (dilation_exp - 1))
            else:
                rf_size_i = 1 + i * (kernel_size - 1)
            t_len_i = self.total_t_len - rf_size_i + 1
            self.temporal_convs.append(TConv(hidden_dim, hidden_dim, kernel_set=kernel_sets, dilation_factor=new_dilation, dropout=dropout, conv_type="1D"))

            self.skip_convs.append(
                torch.nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=t_len_i))

            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
            new_dilation *= dilation_exp

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.start_conv(x)

        if self.is_skip:
           skip = self.skip0(F.dropout(x, self.dropout, training=self.training))

        for i in range(self.n_layers):
            if self.is_residual:
                residual = x
            x = self.temporal_convs[i](x, is_filter=False)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.is_skip:
                skip = skip + self.skip_convs[i](x)

            x = self.bns[i](x)
            x = torch.relu(x)
            if self.is_residual:
                x = x + residual[:, :, -x.size(2):]

        if self.is_skip:
            skip =self.skipE(x)+skip
        else:
            skip = self.skipE(x)

        return skip.squeeze(-1).unsqueeze(1)


# if __name__ == '__main__':
#     # generate a tensor with shape (batch_size=128, seq_len=100, feature_dim=51)
#     x = torch.randn(128, 100, 51)
#     # generate an adjacency matrix
#     adj = torch.randn(128, 51, 51)
#     feature_encoder = FeatureEncoder(51, 100, win_size=5, hidden_dim=64, n_layers=3, topK=15)
#     output = feature_encoder(x)