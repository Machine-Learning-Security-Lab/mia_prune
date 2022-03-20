import argparse
import json
import numpy as np
import os
import pickle
import shutil
import random
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from datasets import get_dataset
from utils import seed_worker

parser = argparse.ArgumentParser()
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='mnist', type=str)
parser.add_argument('--model_name', default='mnist', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--early_stop', default=5, type=int, help="patience for early stopping")
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="adam", type=str)
parser.add_argument('--shadow_num', default=5, type=int)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    save_folder = f"results/{args.dataset_name}_{args.model_name}"

    print(f"Save Folder: {save_folder}")
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder}/data_index.pkl"

    # split the dataset into victim dataset and shadow dataset, then split each into train, val, test
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    victim_list, attack_list = train_test_split(list(range(total_size)), test_size=0.5, random_state=args.seed)
    victim_train_list, victim_test_list = train_test_split(victim_list, test_size=0.45, random_state=args.seed)
    victim_train_list, victim_dev_list = train_test_split(
        victim_train_list, test_size=0.1818, random_state=args.seed)
    attack_split_list = []
    for i in range(args.shadow_num):
        attack_train_list, attack_test_list = train_test_split(
            attack_list, test_size=0.45, random_state=args.seed + i)
        attack_train_list, attack_dev_list = train_test_split(
            attack_train_list, test_size=0.1818, random_state=args.seed + i)
        attack_split_list.append([attack_train_list, attack_dev_list, attack_test_list])
    with open(data_path, 'wb') as f:
        pickle.dump([victim_train_list, victim_dev_list, victim_test_list, attack_split_list], f)

    # train and prune the victim model
    victim_train_dataset = Subset(total_dataset, victim_train_list)
    victim_dev_dataset = Subset(total_dataset, victim_dev_list)
    victim_test_dataset = Subset(total_dataset, victim_test_list)

    print(f"Total Data Size: {total_size}, "
          f"Victim Train Size: {len(victim_train_list)}, "
          f"Victim Dev Size: {len(victim_dev_list)}, "
          f"Victim Test Size: {len(victim_test_list)}")
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)
    victim_dev_loader = DataLoader(victim_dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True, worker_init_fn=seed_worker)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True, worker_init_fn=seed_worker)

    victim_model_save_folder = save_folder + "/victim_model"

    print("Train Victim Model")
    if not os.path.exists(victim_model_save_folder):
        os.makedirs(victim_model_save_folder)
    victim_model = BaseModel(
        args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=victim_model_save_folder,
        device=device, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    best_acc = 0
    count = 0
    for epoch in range(args.epochs):
        train_acc, train_loss = victim_model.train(victim_train_loader, f"Epoch {epoch} Train")
        dev_acc, dev_loss = victim_model.test(victim_dev_loader, f"Epoch {epoch} Dev")
        test_acc, test_loss = victim_model.test(victim_test_loader, f"Epoch {epoch} Test")
        if dev_acc > best_acc:
            best_acc = dev_acc
            save_path = victim_model.save(epoch, test_acc, test_loss)
            best_path = save_path
            count = 0
        elif args.early_stop > 0:
            count += 1
            if count > args.early_stop:
                print(f"Early Stop at Epoch {epoch}")
                break
    shutil.copyfile(best_path, f"{victim_model_save_folder}/best.pth")

    # Train shadow models
    for shadow_ind in range(args.shadow_num):
        attack_train_list, attack_dev_list, attack_test_list = attack_split_list[shadow_ind]
        attack_train_dataset = Subset(total_dataset, attack_train_list)
        attack_dev_dataset = Subset(total_dataset, attack_dev_list)
        attack_test_dataset = Subset(total_dataset, attack_test_list)
        attack_train_loader = DataLoader(attack_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                         pin_memory=True, worker_init_fn=seed_worker)
        attack_dev_loader = DataLoader(attack_dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                       pin_memory=True, worker_init_fn=seed_worker)
        attack_test_loader = DataLoader(attack_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                        pin_memory=True, worker_init_fn=seed_worker)

        print(f"Train Shadow Model {shadow_ind}")
        shadow_model_save_folder = f"{save_folder}/shadow_model_{shadow_ind}"
        if not os.path.exists(shadow_model_save_folder):
            os.makedirs(shadow_model_save_folder)
        shadow_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=shadow_model_save_folder,
            device=device, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
        best_acc = 0
        count = 0
        for epoch in range(args.epochs):
            train_acc, train_loss = shadow_model.train(attack_train_loader, f"Epoch {epoch} Shadow Train")
            dev_acc, dev_loss = shadow_model.test(attack_dev_loader, f"Epoch {epoch} Shadow Dev")
            test_acc, test_loss = shadow_model.test(attack_test_loader, f"Epoch {epoch} Shadow Test")
            if dev_acc > best_acc:
                best_acc = dev_acc
                save_path = shadow_model.save(epoch, test_acc, test_loss)
                best_path = save_path
                count = 0
            elif args.early_stop > 0:
                count += 1
                if count > args.early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break

        shutil.copyfile(best_path, f"{shadow_model_save_folder}/best.pth")


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
        args.prune_epochs = int(args.epochs) // 2

    print(args)
    main(args)
