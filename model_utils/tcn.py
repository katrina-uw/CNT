from torch import nn
import torch.nn.functional as F
import torch

class Dilated_Inception(nn.Module):

    def __init__(self, cin, cout, kernel_set=(2, 3, 5, 7), dilation_factor=2, conv_type="2D"):
        super(Dilated_Inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            if conv_type == "2D":
                self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))
            else:
                self.tconv.append(nn.Conv1d(cin,cout,kern,dilation=dilation_factor))

    def forward(self,input):
        # input [B, D, N, T]
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(-1):]
        x = torch.cat(x, dim=1)
        return x


class TConv(torch.nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, dropout:float, conv_type:str):
        super(TConv, self).__init__()
        self.filter_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor, conv_type)
        self.gate_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor, conv_type)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, is_filter=True):
        _filter = self.filter_conv(x)
        if is_filter:
            filter = torch.tanh(_filter)
            _gate = self.gate_conv(x)
            gate = torch.sigmoid(_gate)
            x = filter * gate
        else:
            x = _filter
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class nconv(torch.nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        """
         :param X: tensor, [B, D, N, T]
         :param A: tensor [N, N] , [B, N, N] or [T*, B, N, N]
         :return: tensor [B, D, N, T]
        """
        #x = torch.einsum('ncwl,vw->ncvl',(x,A))
        if len(A.shape) == 2:
            a_ ='vw'
        elif len(A.shape) == 3:
            a_ ='bvw'
        else:
            a_ = 'tbvw'
        x = torch.einsum(f'bcwt,{a_}->bcvt',(x,A))
        return x.contiguous()


class linear(torch.nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class MixProp(torch.nn.Module):
    def __init__(self, c_in, c_out, gdep, alpha=0.05):
        super(MixProp, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x, adj):
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj)
            out.append(h)
        ho = torch.cat(out, dim=1)  # [B, D*(1+gdep), N, T]
        ho = self.mlp(ho)  # [B, c_out, N, T]
        return ho