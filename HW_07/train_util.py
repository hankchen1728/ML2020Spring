import os
# import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import utils


def _save_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Evaluator(object):
    def __init__(self,
                 net,
                 optimizer,
                 device="cuda",
                 verbose=False):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        self.verbose = verbose
        if verbose:
            self.progress_bar = utils.ProgressBar()
        else:
            self.progress_bar = None

    def run_epoch(self, data_loader, mode="eval"):
        if mode == "eval":
            self.net.eval()
        elif mode == "train":
            self.net.train()
        else:
            return

        num_iters = len(data_loader)
        # Run iteration
        eval_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if mode == "train":
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            elif mode == "eval":
                # Save memory usage
                with torch.no_grad():
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)

            # Summing up the loss and correct
            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if mode == "train" and self.verbose:
                # Show training progress
                self.progress_bar.log(
                    batch_idx,
                    num_iters,
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (eval_loss/(batch_idx+1),
                       100.*correct/total, correct, total)
                    )

        accuracy = 100. * correct / total
        eval_loss /= num_iters
        return accuracy, eval_loss

    def predict(self, data_loader):
        self.net.eval()
        pred_label = np.array([])
        with torch.no_grad():
            for batch_idx, inputs in enumerate(data_loader):
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)

                _, predicted = outputs.max(1)
                pred_label = np.concatenate([pred_label,
                                             predicted.cpu().data.numpy()])
        return pred_label


def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # Original Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # KL Divergence
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(outputs/T, dim=1),
        F.softmax(teacher_outputs/T, dim=1)
        ) * (alpha * T * T)
    return hard_loss + soft_loss


class KD_Evaluator(object):
    def __init__(self,
                 net,
                 optimizer,
                 device="cuda",
                 verbose=False):
        self.net = net
        # Config teacher net
        self.teacher_net = models.resnet18(
            pretrained=False,
            num_classes=11).to(device)
        # if device == "cuda":
        #     self.teacher_net = nn.DataParallel(self.teacher_net)
        self.teacher_net.load_state_dict(
            torch.load(f'./checkpoints/teacher_resnet18_from_scratch.bin')
            )
        # self.teacher_net.load_state_dict(
        #     torch.load(f'./checkpoints/teacher_resnet18.bin')
        #     )
        self.teacher_net.eval()

        self.criterion = loss_fn_kd
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        if verbose:
            self.progress_bar = utils.ProgressBar()
        else:
            self.progress_bar = None

    def run_epoch(self, data_loader, mode="eval"):
        if mode == "eval":
            self.net.eval()
        elif mode == "train":
            self.net.train()

        num_iters = len(data_loader)
        # Run iteration
        eval_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, hard_labels) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            hard_labels = hard_labels.to(self.device)
            # hard_labels = torch.LongTensor(hard_labels).to(self.device)
            # Run teacher net inference
            with torch.no_grad():
                soft_labels = self.teacher_net(inputs)

            if mode == "train":
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, hard_labels, soft_labels)
                loss.backward()
                self.optimizer.step()
            elif mode == "eval":
                # Save memory usage
                with torch.no_grad():
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, hard_labels, soft_labels)

            # Summing up the loss and correct
            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()

            if mode == "train":
                # Show training progress
                self.progress_bar.log(
                    batch_idx,
                    num_iters,
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (eval_loss/(batch_idx+1),
                       100.*correct/total, correct, total)
                    )

        accuracy = 100. * correct / total
        eval_loss /= num_iters
        return accuracy, eval_loss
