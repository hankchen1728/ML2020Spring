import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils


def _save_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_highest_confidence(
        labels, confidences, num_select=2500, target_label=0):
    """Get the indices with highest confidence score w.r.t. selected label
    """
    target_confidence = confidences.copy()
    target_confidence[labels != target_label] = float("-inf")
    target_indices = np.argpartition(
        target_confidence, kth=-num_select
        )[-num_select:]

    return target_indices


class DaNN_Evaluator(object):
    """Evaluator for Autoencoder-based models"""
    def __init__(self,
                 feature_extractor,
                 label_predictor,
                 domain_classifier,
                 device="cuda",
                 verbose=False):
        # Networks
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier = domain_classifier

        self.FE_optimizer = None
        self.LP_optimizer = None
        self.DC_optimizer = None
        # Training criterion
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.class_criterion = nn.CrossEntropyLoss()
        self.device = device
        self.epoch = 0
        self.verbose = verbose
        # Show the progress bar in terminal
        if verbose:
            self.progress_bar = utils.ProgressBar(bar_length=40.)
        else:
            self.progress_bar = None
        # end

    def set_optimizer(self, opt_name="adam", lr=1e-3):
        if opt_name == "adam":
            self.FE_optimizer = optim.Adam(
                self.feature_extractor.parameters(), lr)
            self.LP_optimizer = optim.Adam(
                self.label_predictor.parameters(), lr)
            self.DC_optimizer = optim.Adam(
                self.domain_classifier.parameters(), lr)
        elif opt_name == "sgd":
            self.FE_optimizer = optim.SGD(
                self.feature_extractor.parameters(), lr, momentum=0.9)
            self.LP_optimizer = optim.SGD(
                self.label_predictor.parameters(), lr, momentum=0.9)
            self.DC_optimizer = optim.SGD(
                self.domain_classifier.parameters(), lr, momentum=0.9)
        else:
            print("Unknown optimizer")
        # end

    def finetune_epoch(
            self, source_dataloader, target_dataloader):
        self.epoch += 1
        self.feature_extractor.train()
        self.label_predictor.train()

        num_iters = len(source_dataloader)
        # Run iteration
        mix_loss = 0
        mix_correct, mix_sum = 0, 0
        for batch_idx, (source_data, target_data) in \
                enumerate(zip(source_dataloader, target_dataloader)):
            source_img = source_data[0].to(self.device)
            source_label = source_data[1].to(self.device)
            target_img = target_data[0].to(self.device)
            target_label = target_data[1].to(self.device)

            # Reset the optimizer
            self.FE_optimizer.zero_grad()
            self.LP_optimizer.zero_grad()

            # Mix the source and target dataset
            # num_source, num_target = source_img.shape[0], target_img.shape[0]
            mixed_data = torch.cat([source_img, target_img], dim=0)
            mixed_label = torch.cat([source_label, target_label], dim=0)
            mix_sum += mixed_label.shape[0]

            # Step 1: Training the domain classifier
            feature = self.feature_extractor(mixed_data)
            class_logits = self.label_predictor(feature)
            loss = self.class_criterion(class_logits, mixed_label)
            mix_loss += loss.item()
            loss.backward()
            self.FE_optimizer.step()
            self.LP_optimizer.step()

            mix_correct += torch.sum(
                torch.argmax(class_logits, dim=1) == mixed_label
                ).item()

            # Show training progress
            if self.verbose:
                self.progress_bar.log(
                    batch_idx,
                    num_iters,
                    "Epoch %03d | Classifier Loss: %.5f | Acc: %.4f"
                    % (self.epoch, mix_loss/(batch_idx+1),
                       mix_correct/mix_sum)
                )

        mix_loss /= num_iters
        mix_acc = mix_correct / mix_sum
        return mix_loss, mix_acc

    def train_epoch(
            self, source_dataloader, target_dataloader,
            loss_lambda=0.1):
        self.epoch += 1
        self.feature_extractor.train()
        self.label_predictor.train()
        self.domain_classifier.train()

        num_iters = min(len(source_dataloader), len(target_dataloader))
        # Run iteration
        domain_loss = 0
        mix_loss = 0
        source_correct, source_sum = 0, 0
        for batch_idx, ((source_data, source_label), (target_data, _)) in \
                enumerate(zip(source_dataloader, target_dataloader)):
            source_data = source_data.to(self.device)
            source_label = source_label.to(self.device)
            target_data = target_data.to(self.device)

            # Reset the optimizer
            self.FE_optimizer.zero_grad()
            self.LP_optimizer.zero_grad()
            self.DC_optimizer.zero_grad()

            # Mix the source and target dataset
            num_source, num_target = source_data.shape[0], target_data.shape[0]
            mixed_data = torch.cat([source_data, target_data], dim=0)
            # Setting the domain label
            domain_label = torch.zeros(
                [num_source + num_target, 1]
            ).to(self.device)
            domain_label[:num_source] = 1

            # Step 1: Training the domain classifier
            feature = self.feature_extractor(mixed_data)
            domain_logits = self.domain_classifier(feature.detach())
            loss = self.domain_criterion(domain_logits, domain_label)
            domain_loss += loss.item()
            loss.backward()
            self.DC_optimizer.step()

            # Step 2: Training feature extractor and label preditor
            class_logits = self.label_predictor(feature[:num_source])
            domain_logits = self.domain_classifier(feature)
            loss = self.class_criterion(class_logits, source_label) - \
                loss_lambda * \
                self.domain_criterion(domain_logits, domain_label)
            mix_loss += loss.item()
            loss.backward()
            self.FE_optimizer.step()
            self.LP_optimizer.step()

            source_correct += torch.sum(
                torch.argmax(class_logits, dim=1) == source_label
                ).item()
            source_sum += num_source

            # Show training progress
            if self.verbose:
                self.progress_bar.log(
                    batch_idx,
                    num_iters,
                    "Epoch %03d | Domain Loss: %.5f | "
                    "Mixed Loss: %.5f | Source Acc: %.4f"
                    % (self.epoch, domain_loss/(batch_idx+1),
                       mix_loss/(batch_idx+1),
                       source_correct/source_sum)
                )

        domain_loss /= num_iters
        mix_loss /= num_iters
        source_acc = source_correct / source_sum
        return domain_loss, mix_loss, source_acc

    def pred_acc(self, dataloader):
        self.feature_extractor.eval()
        self.label_predictor.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (img_data, img_label) in enumerate(dataloader):
                img_data = img_data.to(self.device)
                img_label = img_label.to(self.device)
                class_logits = self.label_predictor(
                    self.feature_extractor(img_data))
                correct += torch.sum(
                    torch.argmax(class_logits, dim=1) == img_label
                    ).item()
                total += img_label.shape[0]
        return correct / total
