import torch
import torch.nn as nn
from dataloader import ImageDataset, Dataloader


class DNN_Model(nn.Module):

    def __init__(self):
        super(DNN_Model, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        return x


def save_model(model, optimizer, store_model_path):
    # save model and optimizer
    torch.save(model.state_dict(), f'{store_model_path}.ckpt')
    torch.save(optimizer.state_dict(), f'{store_model_path}.opt')
    return


def load_model(model, optimizer, load_model_path):
    # load model and optimizer
    print(f'Load model from {load_model_path}')
    model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
    optimizer.load_state_dict(torch.load(f'{load_model_path}.opt'))
    return model, optimizer


def build_model(data_path, batch_size, learning_rate, device="cuda"):
    # create model
    model = DNN_Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data = ImageDataset(data_path)
    datasets = data.get_datasets()
    tasks = []
    for dataset in datasets:
        tasks.append(Dataloader(dataset, batch_size))

    return model, optimizer, tasks
