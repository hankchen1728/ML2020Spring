import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class WGenerator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(WGenerator, self).__init__()

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(in_dim, dim*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dim*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(dim*8, dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(dim*2, dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(dim, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = x.view(x.size(0), -1, 1, 1)  # (batch size, input dim, 1, 1)
        y = self.cnn(y)
        return y


class WDiscriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(WDiscriminator, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim, dim, 4, 2, 1, bias=False),  # (N, dim, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False),  # (N, dim*2, 16, 16)
            nn.BatchNorm2d(dim*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=False),  # (N, dim*4, 8, 8)
            nn.BatchNorm2d(dim*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim*4, dim*8, 4, 2, 1, bias=False),  # (N, dim*8, 4, 4)
            nn.BatchNorm2d(dim*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim*8, 1, 4, 1, 0, bias=False),  # (N, 1, 1, 1)
            # Modification 1: remove sigmoid
            # nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.cnn(x)
        y = y.view(-1)
        return y
