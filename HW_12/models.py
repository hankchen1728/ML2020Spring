import torch.nn as nn


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 64, 3, 1, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),

        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),

        #     nn.MaxPool2d(2),

        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),

        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        vgg11_cfg = \
            [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = make_layers(vgg11_cfg, batch_norm=True)

    def forward(self, x):
        x = self.features(x).squeeze()
        return x


class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 10),
        )

    def forward(self, h):
        c = self.classifier(h)
        return c


class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),

            nn.Linear(128, 1),
        )

    def forward(self, h):
        y = self.classifier(h)
        return y
