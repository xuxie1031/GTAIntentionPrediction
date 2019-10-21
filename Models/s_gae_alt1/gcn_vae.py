import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super(GraphConvS, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              dilation=dilation,
                              bias=bias)

    # x: (N, C, V, V); A: (N, V, V)
    def forward(self, x, A):
        assert x.size(2) == x.size(3) == A.size(1)

        x = self.conv(x)

        x = torch.einsum('ncuv,nvw->ncuw', (x, A))

        return x.contiguous(), A


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout=0.0):
        super(InnerProductDecoder, self).__init__()

        self.dropout = dropout

    # z: (N, V, C)
    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        
        z_t = z.permute(0, 2, 1).contiguous()
        A = torch.sigmoid(torch.einsum('nuc,ncv->nuv', (z, z_t)))

        return A


class GCNVAE(nn.Module):
    def __init__(self, in_channels, h_dim1, h_dim2, dropout=0.0, use_cuda=True, device=None):
        super(GCNVAE, self).__init__()

        self.dropout = dropout
        self.gc0 = GraphConvS(in_channels, h_dim1)
        self.gc1 = GraphConvS(h_dim1, h_dim1)
        self.gc2 = GraphConvS(h_dim1, h_dim2)
        self.gc3 = GraphConvS(h_dim1, h_dim2)
        self.dc = InnerProductDecoder(dropout=dropout)
        self.bn = nn.BatchNorm2d(h_dim1)
        
        if use_cuda:
            self.to(device)


    def encode(self, x, A):
        hidden, _ = self.gc0(x, A)
        hidden = self.bn(hidden)
        hidden = F.dropout(hidden, self.dropout, self.training)
        hidden = F.relu(hidden)

        hidden, _ = self.gc1(hidden, A)
        hidden = self.bn(hidden)
        hidden = F.dropout(hidden, self.dropout, self.training)
        hidden = F.relu(hidden)

        mu, _ = self.gc2(hidden, A)
        logvar, _ = self.gc3(hidden, A)

        N, C, U, V = mu.size()
        
        mu = mu.mean(dim=-1)
        mu = mu.permute(0, 2, 1).contiguous()

        logvar = logvar.mean(dim=-1)
        logvar = logvar.permute(0, 2, 1).contiguous()

        return mu, logvar


    def parameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    
    def forward(self, x, A):
        mu, logvar = self.encode(x, A)
        z = self.parameterize(mu, logvar)

        return self.dc(z), mu, logvar