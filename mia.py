import argparse
import json
import numpy as np
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from datasets import get_dataset
from attackers import MiaAttack

parser = argparse.ArgumentParser(description='Membership inference Attacks on Network Pruning')
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='mnist', type=str)
parser.add_argument('--model_name', default='mnist', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=1, type=int)
parser.add_argument('--image_size', default=28, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--early_stop', default=5, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--prune_epochs', default=50, type=int)
parser.add_argument('--pruner_name', default='l1unstructure', type=str, help="prune method for victim model")
parser.add_argument('--prune_sparsity', default=0.7, type=float, help="prune sparsity for victim model")
parser.add_argument('--adaptive', action='store_true', help="use adaptive attack")
parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--defend', default='', type=str)
parser.add_argument('--defend_arg', default=4, type=float)
parser.add_argument('--attacks', default="samia", type=str)
parser.add_argument('--original', action='store_true', help="original=true, then launch attack against original model")


def main(args):
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    prune_prefix = f"{args.pruner_name}_{args.prune_sparsity}" \
                   f"{'_' + args.defend if args.defend else ''}{'_' + str(args.defend_arg) if args.defend else ''}"
    prune_prefix2 = f"{args.pruner_name}_{args.prune_sparsity}" \
                    f"{'_' + args.defend if args.adaptive else ''}{'_' + str(args.defend_arg) if args.adaptive else ''}"

    save_folder = f"results/{args.dataset_name}_{args.model_name}"

    print(f"Save Folder: {save_folder}")

    # Load datasets
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder}/data_index.pkl"
    with open(data_path, 'rb') as f:
        victim_train_list, victim_dev_list, victim_test_list, attack_split_list \
            = pickle.load(f)
    victim_train_dataset = Subset(total_dataset, victim_train_list)
    victim_test_dataset = Subset(total_dataset, victim_test_list)
    print(f"Total Data Size: {total_size}, "
          f"Victim Train Size: {len(victim_train_list)}, "
          f"Victim Test Size: {len(victim_test_list)}")
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    # Load pruned victim model
    victim_model_save_folder = save_folder + "/victim_model"
    victim_model_path = f"{victim_model_save_folder}/best.pth"
    victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path)

    pruned_model_save_folder = f"{save_folder}/{prune_prefix}_model"
    print(f"Load Pruned Model from {pruned_model_save_folder}")
    victim_pruned_model = BaseModel(
        args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=pruned_model_save_folder,
        device=device)
    victim_pruned_model.model.load_state_dict(torch.load(f"{pruned_model_save_folder}/best.pth"))
    victim_pruned_model.test(victim_train_loader, "Victim Pruned Model Train")
    victim_pruned_model.test(victim_test_loader, "Victim Pruned Model Test")

    # Load pruned shadow models
    shadow_model_list, shadow_prune_model_list, shadow_train_loader_list, shadow_test_loader_list = [], [], [], []
    for shadow_ind in range(args.shadow_num):
        attack_train_list, attack_dev_list, attack_test_list = attack_split_list[shadow_ind]
        shadow_train_dataset = Subset(total_dataset, attack_train_list)
        shadow_dev_dataset = Subset(total_dataset, attack_dev_list)
        shadow_test_dataset = Subset(total_dataset, attack_test_list)
        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         pin_memory=False)
        shadow_dev_loader = DataLoader(shadow_dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                       pin_memory=False)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                        pin_memory=False)

        shadow_model_path = f"{save_folder}/shadow_model_{shadow_ind}/best.pth"
        shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        shadow_model.load(shadow_model_path)
        pruned_shadow_model_save_folder = f"{save_folder}/shadow_{prune_prefix2}_model_{shadow_ind}"
        print(f"Load Pruned Shadow Model From {pruned_shadow_model_save_folder}")
        shadow_pruned_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim,
            save_folder=pruned_shadow_model_save_folder, device=device)
        shadow_pruned_model.model.load_state_dict(torch.load(f"{pruned_shadow_model_save_folder}/best.pth"))
        shadow_pruned_model.test(shadow_train_loader, "Shadow Pruned Model Train")
        shadow_pruned_model.test(shadow_test_loader, "Shadow Pruned Model Test")

        shadow_model_list.append(shadow_model)
        shadow_prune_model_list.append(shadow_pruned_model)
        shadow_train_loader_list.append(shadow_train_loader)
        shadow_test_loader_list.append(shadow_test_loader)

    print("Start Membership Inference Attacks")

    if args.original:
        attack_original = True
    else:
        attack_original = False
    attacker = MiaAttack(
        victim_model, victim_pruned_model, victim_train_loader, victim_test_loader,
        shadow_model_list, shadow_prune_model_list, shadow_train_loader_list, shadow_test_loader_list,
        num_cls=args.num_cls, device=device, batch_size=args.batch_size,
        attack_original=attack_original)

    attacks = args.attacks.split(',')

    if "samia" in attacks:
        nn_trans_acc = attacker.nn_attack("nn_sens_cls", model_name="transformer")
        print(f"SAMIA attack accuracy {nn_trans_acc:.3f}")

    if "threshold" in attacks:
        conf, xent, mentr, top1_conf = attacker.threshold_attack()
        print(f"Ground-truth class confidence-based threshold attack (Conf) accuracy: {conf:.3f}")
        print(f"Cross-entropy-based threshold attack (Xent) accuracy: {xent:.3f}")
        print(f"Modified-entropy-based threshold attack (Mentr) accuracy: {mentr:.3f}")
        print(f"Top1 Confidence-based threshold attack (Top1-conf) accuracy: {top1_conf:.3f}")

    if "nn" in attacks:
        nn_acc = attacker.nn_attack("nn")
        print(f"NN attack accuracy {nn_acc:.3f}")

    if "nn_top3" in attacks:
        nn_top3_acc = attacker.nn_attack("nn_top3")
        print(f"Top3-NN Attack Accuracy {nn_top3_acc}")

    if "nn_cls" in attacks:
        nn_cls_acc = attacker.nn_attack("nn_cls")
        print(f"NNCls Attack Accuracy {nn_cls_acc}")


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    print(args)
    main(args)
