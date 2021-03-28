import os
import argparse
import json
import torch

from model_utils import save_model, load_model, build_model
from lifelong import EWC, MAS, SCP
from train_utils import normal_train, ewc_train, mas_train, scp_train


def val(model, task, device="cuda"):
    model.eval()
    correct_cnt = 0
    for imgs, labels in task.val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred_label = torch.max(outputs.data, 1)

        correct_cnt += (pred_label == labels.data).sum().item()

    return correct_cnt / task.val_dataset_size


def train_process(model, optimizer, tasks, config, device="cuda"):
    task_loss, acc = {}, {}
    for task_id, task in enumerate(tasks):
        print('\n')
        total_epochs = 0
        task_loss[task.name] = []
        acc[task.name] = []

        # Part 1: normal, EWC, MAS
        if config.mode == 'basic' or task_id == 0:
            while (total_epochs < config.num_epochs):
                model, optimizer, losses = normal_train(
                    model, optimizer, task,
                    total_epochs,
                    config.summary_epochs,
                    device=device)
                task_loss[task.name] += losses

                for subtask in range(task_id + 1):
                    acc[tasks[subtask].name].append(val(model, tasks[subtask]))

                total_epochs += config.summary_epochs
                if total_epochs % config.store_epochs == 0 or \
                        total_epochs >= config.num_epochs:
                    save_model(model, optimizer, config.store_model_path)

        if config.mode == 'ewc' and task_id > 0:
            old_dataloaders = []
            for old_task in range(task_id):
                old_dataloaders += [tasks[old_task].val_loader]
            ewc = EWC(model, old_dataloaders, device)
            while (total_epochs < config.num_epochs):
                model, optimizer, losses = ewc_train(
                    model, optimizer, task,
                    total_epochs,
                    config.summary_epochs, ewc,
                    config.lifelong_coeff,
                    device=device)
                task_loss[task.name] += losses

                for subtask in range(task_id + 1):
                    acc[tasks[subtask].name].append(val(model, tasks[subtask]))

                total_epochs += config.summary_epochs
                if total_epochs % config.store_epochs == 0 or \
                        total_epochs >= config.num_epochs:
                    save_model(model, optimizer, config.store_model_path)

        if config.mode == 'mas' and task_id > 0:
            old_dataloaders = []
            mas_tasks = []
            for old_task in range(task_id):
                old_dataloaders += [tasks[old_task].val_loader]
                mas = MAS(model, old_dataloaders, device)
                mas_tasks += [mas]
            while (total_epochs < config.num_epochs):
                model, optimizer, losses = mas_train(
                    model, optimizer, task,
                    total_epochs,
                    config.summary_epochs,
                    mas_tasks,
                    config.lifelong_coeff,
                    device=device)
                task_loss[task.name] += losses

                for subtask in range(task_id + 1):
                    acc[tasks[subtask].name].append(val(model, tasks[subtask]))

                total_epochs += config.summary_epochs
                if total_epochs % config.store_epochs == 0 or \
                        total_epochs >= config.num_epochs:
                    save_model(model, optimizer, config.store_model_path)

        # Part2 : SCP
        if config.mode == 'scp' and task_id > 0:
            old_dataloaders = []
            scp_tasks = []
            for old_task in range(task_id):
                old_dataloaders += [tasks[old_task].val_loader]
                scp = SCP(model, old_dataloaders, 100, device)
                scp_tasks += [scp]
            while (total_epochs < config.num_epochs):
                model, optimizer, losses = scp_train(
                    model, optimizer, task,
                    total_epochs,
                    config.summary_epochs,
                    scp_tasks,
                    config.lifelong_coeff,
                    device=device)
                task_loss[task.name] += losses

                for subtask in range(task_id + 1):
                    acc[tasks[subtask].name].append(val(model, tasks[subtask]))

                total_epochs += config.summary_epochs
                if total_epochs % config.store_epochs == 0 or \
                        total_epochs >= config.num_epochs:
                    save_model(model, optimizer, config.store_model_path)
    return task_loss, acc


class configurations(object):
    def __init__(self):
        self.batch_size = 256
        self.num_epochs = 10000
        self.store_epochs = 250
        self.summary_epochs = 250
        self.learning_rate = 0.0005
        self.load_model = False
        self.store_model_path = "./model"
        self.load_model_path = "./model"
        self.data_path = "./data"
        self.mode = None
        self.lifelong_coeff = 0.5


def main(args):
    # Set CUDA GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = \
        ','.join(str(gpu) for gpu in args.visible_gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode_list = args.mode_list

    coeff_list = args.coeff_list

    assert len(mode_list) == len(coeff_list)

    config = configurations()
    count = 0
    for mode in mode_list:
        config.mode = mode
        config.lifelong_coeff = coeff_list[count]
        lifelong_coeff = coeff_list[count]
        print("{} training".format(config.mode))
        model, optimizer, tasks = build_model(
            config.data_path, config.batch_size, config.learning_rate)
        print("Finish build model")
        if config.load_model:
            model, optimizer = load_model(
                model, optimizer, config.load_model_path)
        task_loss, acc = train_process(model, optimizer, tasks, config, device)
        with open(f'./{config.mode}_{lifelong_coeff}_acc.txt', 'w') as f:
            json.dump(acc, f)
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Lifelong Training"
    )

    parser.add_argument(
        "--visible_gpus",
        type=int,
        nargs='+',
        default=[0],
        help="CUDA visible gpus")

    parser.add_argument(
        "--mode_list",
        type=str,
        nargs='+',
        default=["basic"],
        help="Life long modes")

    parser.add_argument(
        "--coeff_list",
        type=float,
        nargs='+',
        default=[0],
        help="Life long coeff")

    args = parser.parse_args()

    main(args)
