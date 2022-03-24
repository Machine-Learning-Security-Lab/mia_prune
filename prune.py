import argparse
import copy
import json
import numpy as np
import os
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from datasets import get_dataset
from pruner import get_pruner
from utils import seed_worker
from pyvacy import optim, analysis

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
parser.add_argument('--prune_epochs', default=50, type=int)
parser.add_argument('--pruner_name', default='l1unstructure', type=str)
parser.add_argument('--prune_sparsity', default=0.7, type=float)
parser.add_argument('--defend', default="", type=str, help="'' if no defense, else ppb")
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--defend_arg', default=4, type=float)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    prune_lr = args.lr
    if args.defend == "":
        prune_prefix = f"{args.pruner_name}_{args.prune_sparsity}"
    else:
        prune_prefix = f"{args.pruner_name}_{args.prune_sparsity}_{args.defend}_{args.defend_arg}"

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

    # load data split for the pretrained victim and shadow model
    with open(data_path, 'rb') as f:
        victim_train_list, victim_dev_list, victim_test_list, attack_split_list \
            = pickle.load(f)

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
    # load pretrained model
    victim_model_path = f"{victim_model_save_folder}/best.pth"
    victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path)
    test_acc, test_loss = victim_model.test(victim_test_loader, "Pretrained Victim")

    victim_acc = test_acc

    print("Prune Victim Model")
    pruned_model_save_folder = f"{save_folder}/{prune_prefix}_model"
    victim_model_path = f"{victim_model_save_folder}/best.pth"
    victim_model.load(victim_model_path)

    org_state = copy.deepcopy(victim_model.model.state_dict())
    if not os.path.exists(pruned_model_save_folder):
        os.makedirs(pruned_model_save_folder)

    # prune victim model
    if args.defend == "adv":
        attack_model_type = "mia_fc"
    else:
        attack_model_type = ""

    victim_pruned_model = BaseModel(
        args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, lr=prune_lr,
        weight_decay=args.weight_decay, save_folder=pruned_model_save_folder, device=device,
        optimizer=args.optimizer, attack_model_type=attack_model_type)
    victim_pruned_model.model.load_state_dict(org_state)
    pruner = get_pruner(args.pruner_name, victim_pruned_model.model, sparsity=args.prune_sparsity)
    victim_pruned_model.model = pruner.compress()

    best_acc = 0
    count = 0
    for epoch in range(args.prune_epochs):
        pruner.update_epoch(epoch)
        if args.defend == "":
            train_acc, train_loss = victim_pruned_model.train(victim_train_loader, f"Epoch {epoch} Prune Train")
        elif args.defend == "ppb":
            train_acc, train_loss = victim_pruned_model.train_defend_ppb(
                victim_train_loader, log_pref=f"Epoch {epoch} Victim Prune Train With PPB", defend_arg=args.defend_arg)
        elif args.defend == "adv":
            train_acc, train_loss = victim_pruned_model.train_defend_adv(
                victim_train_loader, victim_dev_loader, log_pref=f"Epoch {epoch} Victim Prune Train With ADV",
                privacy_theta=args.defend_arg)
        dev_acc, dev_loss = victim_pruned_model.test(victim_dev_loader, f"Epoch {epoch} Prune Dev")
        test_acc, test_loss = victim_pruned_model.test(victim_test_loader, f"Epoch {epoch} Prune Test")

        if dev_acc > best_acc:
            best_acc = dev_acc
            pruner.export_model(model_path=f"{pruned_model_save_folder}/best.pth",
                                mask_path=f"{pruned_model_save_folder}/best_mask.pth")
            count = 0
        elif args.early_stop > 0:
            count += 1
            if count > args.early_stop:
                print(f"Early Stop at Epoch {epoch}")
                break

    victim_prune_acc = test_acc

    # prune shadow models
    shadow_acc_list = []
    shadow_prune_acc_list = []
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

        # load pretrained shadow model
        shadow_model_path = f"{save_folder}/shadow_model_{shadow_ind}/best.pth"
        shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        shadow_model.load(shadow_model_path)
        test_acc, _ = shadow_model.test(attack_test_loader, f"Pretrain Shadow")
        shadow_acc = test_acc

        org_state = copy.deepcopy(shadow_model.model.state_dict())
        pruned_shadow_model_save_folder = \
            f"{save_folder}/shadow_{prune_prefix}_model_{shadow_ind}"
        if not os.path.exists(pruned_shadow_model_save_folder):
            os.makedirs(pruned_shadow_model_save_folder)

        # prune shadow models
        shadow_pruned_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, lr=prune_lr,
            weight_decay=args.weight_decay, save_folder=pruned_shadow_model_save_folder, device=device,
            optimizer=args.optimizer, attack_model_type=attack_model_type)
        shadow_pruned_model.model.load_state_dict(org_state)
        pruner = get_pruner(args.pruner_name, shadow_pruned_model.model, sparsity=args.prune_sparsity,)
        shadow_pruned_model.model = pruner.compress()

        best_acc = 0
        count = 0
        for epoch in range(args.prune_epochs):
            pruner.update_epoch(epoch)
            if args.defend == "":
                train_acc, train_loss = shadow_pruned_model.train(
                    attack_train_loader, f"Epoch {epoch} Shadow Prune Train")
            elif args.defend == "ppb":
                train_acc, train_loss = shadow_pruned_model.train_defend_ppb(
                    attack_train_loader, f"Epoch {epoch} Shadow Prune Train With PPB", defend_arg=args.defend_arg)
            elif args.defend == "adv":
                train_acc, train_loss = shadow_pruned_model.train_defend_adv(
                    attack_train_loader, attack_dev_loader, log_pref=f"Epoch {epoch} Victim Prune Train With ADV",
                    privacy_theta=args.defend_arg)
            dev_acc, dev_loss = shadow_pruned_model.test(attack_dev_loader, f"Epoch {epoch} Shadow Prune Dev")
            test_acc, test_loss = shadow_pruned_model.test(attack_test_loader, f"Epoch {epoch} Shadow Prune Test")

            if dev_acc > best_acc:
                best_acc = dev_acc
                pruner.export_model(model_path=f"{pruned_shadow_model_save_folder}/best.pth",
                                    mask_path=f"{pruned_shadow_model_save_folder}/best_mask.pth")
                count = 0
            elif args.early_stop > 0:
                count += 1
                if count > args.early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break

        shadow_prune_acc = test_acc
        shadow_acc_list.append(shadow_acc), shadow_prune_acc_list.append(shadow_prune_acc)
    return victim_acc, victim_prune_acc, np.mean(shadow_acc_list), np.mean(shadow_prune_acc_list)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
        args.prune_epochs = int(args.epochs) // 2

    print(args)
    main(args)
