import torch
import torch.nn.functional as F
from attacker_threshold import ThresholdAttacker
from base_model import BaseModel

from torch.utils.data import DataLoader, TensorDataset
from utils import seed_worker


class MiaAttack:
    def __init__(self, victim_model, victim_pruned_model, victim_train_loader, victim_test_loader,
                 shadow_model_list, shadow_pruned_model_list, shadow_train_loader_list, shadow_test_loader_list,
                 num_cls=10, batch_size=128,  device="cuda",
                 lr=0.001, optimizer="sgd", epochs=100, weight_decay=5e-4,
                 # lr=0.001, optimizer="adam", epochs=100, weight_decay=5e-4,
                 attack_original=False
                 ):
        self.victim_model = victim_model
        self.victim_pruned_model = victim_pruned_model
        self.victim_train_loader = victim_train_loader
        self.victim_test_loader = victim_test_loader
        self.shadow_model_list = shadow_model_list
        self.shadow_pruned_model_list = shadow_pruned_model_list
        self.shadow_train_loader_list = shadow_train_loader_list
        self.shadow_test_loader_list = shadow_test_loader_list
        self.num_cls = num_cls
        self.device = device
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.attack_original = attack_original
        self._prepare()

    def _prepare(self):
        attack_in_predicts_list, attack_in_targets_list, attack_in_sens_list = [], [], []
        attack_out_predicts_list, attack_out_targets_list, attack_out_sens_list = [], [], []
        for shadow_model, shadow_pruned_model, shadow_train_loader, shadow_test_loader in zip(
                self.shadow_model_list, self.shadow_pruned_model_list, self.shadow_train_loader_list,
                self.shadow_test_loader_list):

            if self.attack_original:
                attack_in_predicts, attack_in_targets, attack_in_sens = \
                    shadow_model.predict_target_sensitivity(shadow_train_loader)
                attack_out_predicts, attack_out_targets, attack_out_sens = \
                    shadow_model.predict_target_sensitivity(shadow_test_loader)
            else:
                attack_in_predicts, attack_in_targets, attack_in_sens = \
                    shadow_pruned_model.predict_target_sensitivity(shadow_train_loader)
                attack_out_predicts, attack_out_targets, attack_out_sens = \
                    shadow_pruned_model.predict_target_sensitivity(shadow_test_loader)

            attack_in_predicts_list.append(attack_in_predicts)
            attack_in_targets_list.append(attack_in_targets)
            attack_in_sens_list.append(attack_in_sens)
            attack_out_predicts_list.append(attack_out_predicts)
            attack_out_targets_list.append(attack_out_targets)
            attack_out_sens_list.append(attack_out_sens)

        self.attack_in_predicts = torch.cat(attack_in_predicts_list, dim=0)
        self.attack_in_targets = torch.cat(attack_in_targets_list, dim=0)
        self.attack_in_sens = torch.cat(attack_in_sens_list, dim=0)
        self.attack_out_predicts = torch.cat(attack_out_predicts_list, dim=0)
        self.attack_out_targets = torch.cat(attack_out_targets_list, dim=0)
        self.attack_out_sens = torch.cat(attack_out_sens_list, dim=0)

        if self.attack_original:
            self.victim_in_predicts, self.victim_in_targets, self.victim_in_sens = \
                self.victim_model.predict_target_sensitivity(self.victim_train_loader)
            self.victim_out_predicts, self.victim_out_targets, self.victim_out_sens = \
                self.victim_model.predict_target_sensitivity(self.victim_test_loader)
        else:
            self.victim_in_predicts, self.victim_in_targets, self.victim_in_sens = \
                self.victim_pruned_model.predict_target_sensitivity(self.victim_train_loader)
            self.victim_out_predicts, self.victim_out_targets, self.victim_out_sens = \
                self.victim_pruned_model.predict_target_sensitivity(self.victim_test_loader)

    def nn_attack(self, mia_type="nn_sens_cls", model_name="mia_fc"):
        attack_predicts = torch.cat([self.attack_in_predicts, self.attack_out_predicts], dim=0)
        attack_sens = torch.cat([self.attack_in_sens, self.attack_out_sens], dim=0)
        attack_targets = torch.cat([self.attack_in_targets, self.attack_out_targets], dim=0)
        attack_targets = F.one_hot(attack_targets, num_classes=self.num_cls).float()
        attack_labels = torch.cat([torch.ones(self.attack_in_targets.size(0)),
                                   torch.zeros(self.attack_out_targets.size(0))], dim=0).long()

        victim_predicts = torch.cat([self.victim_in_predicts, self.victim_out_predicts], dim=0)
        victim_sens = torch.cat([self.victim_in_sens, self.victim_out_sens], dim=0)
        victim_targets = torch.cat([self.victim_in_targets, self.victim_out_targets], dim=0)
        victim_targets = F.one_hot(victim_targets, num_classes=self.num_cls).float()
        victim_labels = torch.cat([torch.ones(self.victim_in_targets.size(0)),
                                   torch.zeros(self.victim_out_targets.size(0))], dim=0).long()

        if mia_type == "nn_cls":
            new_attack_data = torch.cat([attack_predicts, attack_targets], dim=1)
            new_victim_data = torch.cat([victim_predicts, victim_targets], dim=1)
        elif mia_type == "nn_top3":
            new_attack_data, _ = torch.topk(attack_predicts, k=3, dim=-1)
            new_victim_data, _ = torch.topk(victim_predicts, k=3, dim=-1)
        elif mia_type == "nn_sens_cls":
            new_attack_data = torch.cat([attack_predicts, attack_sens, attack_targets], dim=1)
            new_victim_data = torch.cat([victim_predicts, victim_sens, victim_targets], dim=1)
        else:
            new_attack_data = attack_predicts
            new_victim_data = victim_predicts

        attack_train_dataset = TensorDataset(new_attack_data, attack_labels)
        attack_train_dataloader = DataLoader(
            attack_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)
        attack_test_dataset = TensorDataset(new_victim_data, victim_labels)
        attack_test_dataloader = DataLoader(
            attack_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)

        attack_model = BaseModel(
            model_name, device=self.device, num_cls=new_victim_data.size(1), optimizer=self.optimizer, lr=self.lr,
            weight_decay=self.weight_decay, epochs=self.epochs)

        for epoch in range(self.epochs):
            train_acc, train_loss = attack_model.train(attack_train_dataloader)
            test_acc, test_loss = attack_model.test(attack_test_dataloader)
        return test_acc

    def threshold_attack(self):
        victim_in_predicts = self.victim_in_predicts.numpy()
        victim_out_predicts = self.victim_out_predicts.numpy()

        attack_in_predicts = self.attack_in_predicts.numpy()
        attack_out_predicts = self.attack_out_predicts.numpy()
        attacker = ThresholdAttacker((attack_in_predicts, self.attack_in_targets.numpy()),
                                 (attack_out_predicts, self.attack_out_targets.numpy()),
                                 (victim_in_predicts, self.victim_in_targets.numpy()),
                                 (victim_out_predicts, self.victim_out_targets.numpy()),
                                 self.num_cls)
        confidence, entropy, modified_entropy = attacker._mem_inf_benchmarks()
        top1_conf, _, _ = attacker._mem_inf_benchmarks_non_cls()
        return confidence * 100., entropy * 100., modified_entropy * 100., \
               top1_conf * 100.
