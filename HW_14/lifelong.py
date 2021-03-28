import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC(object):
    """
      @article{kirkpatrick2017overcoming,
          title={Overcoming catastrophic forgetting in neural networks},
          author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz,
                  Neil and Veness, Joel and Desjardins, Guillaume and Rusu,
                  Andrei A and Milan, Kieran and Quan, John and Ramalho,
                  Tiago and Grabska-Barwinska, Agnieszka and others},
          journal={Proceedings of the national academy of sciences},
          year={2017},
          url={https://arxiv.org/abs/1612.00796}
      }
    """

    def __init__(self, model: nn.Module, dataloaders: list, device="cuda"):

        self.model = model
        self.dataloaders = dataloaders
        self.device = device

        self.params = {
            n: p for n,
            p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._calculate_importance()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        # dataloader_num = len(self.dataloaders)
        number_data = sum([len(loader) for loader in self.dataloaders])
        for dataloader in self.dataloaders:
            for data in dataloader:
                self.model.zero_grad()
                input = data[0].to(self.device)
                output = self.model(input).view(1, -1)
                label = output.max(1)[1].view(-1)

                # Fisher matrix for EWC
                loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                loss.backward()

                for n, p in self.model.named_parameters():
                    precision_matrices[n].data += p.grad.data ** 2 / \
                        number_data

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class MAS(object):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny,
              Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/
           Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """

    def __init__(self, model: nn.Module, dataloaders: list, device="cuda"):
        self.model = model
        self.dataloaders = dataloaders
        self.params = {
            n: p for n,
            p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.device = device
        self._precision_matrices = self.calculate_importance()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def calculate_importance(self):
        print("Computing MAS")

        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        # dataloader_num = len(self.dataloaders)
        num_data = sum([len(loader) for loader in self.dataloaders])
        for dataloader in self.dataloaders:
            for data in dataloader:
                self.model.zero_grad()
                output = self.model(data[0].to(self.device))

                # Omega matrix for MAS
                output.pow_(2)
                loss = torch.sum(output, dim=1)
                loss = loss.mean()
                loss.backward()

                for n, p in self.model.named_parameters():
                    # difference with EWC
                    precision_matrices[n].data += p.grad.abs() / num_data

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


class SCP(object):
    """
    OPEN REVIEW VERSION:
    https://openreview.net/forum?id=BJge3TNKwH
    """

    def __init__(self, model: nn.Module, dataloaders: list, L: int, device):
        self.model = model
        self.dataloaders = dataloaders
        self.params = {
            n: p for n,
            p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.L = L
        self.device = device
        self._precision_matrices = self.calculate_importance()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def calculate_importance(self):
        print('Computing SCP')

        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        # dataloader_num = len(self.dataloaders)
        # num_data = sum([len(loader) for loader in self.dataloaders])
        for dataloader in self.dataloaders:
            for data in dataloader:
                self.model.zero_grad()
                output = self.model(data[0].to(self.device))

                # Generate the Gamma matrix for SCP
                # Step1: taking mean of the output
                output_mean = torch.mean(output, dim=0)

                # Step2: Sampling from unit sphere
                kase = sample_spherical(self.L, 10).transpose()
                kase = torch.from_numpy(kase).float().to(self.device)

                # Step3
                for i in range(self.L):
                    lo = torch.dot(kase[i], output_mean)
                    lo.backward(retain_graph=True)
                    for n, p in self.model.named_parameters():
                        precision_matrices[n].data += p.grad.data**2/self.L
                    self.model.zero_grad()

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
