import torch.nn as nn


class AutoEncoder(nn.Module):
    """Auto Encoder in pytorch implementation"""
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconst = self.decoder(encoded)
        return encoded, reconst


class BaselineEncoder(nn.Module):
    """Auto Encoder in pytorch implementation"""
    def __init__(self):
        super(BaselineEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconst = self.decoder(encoded)
        return encoded, reconst
