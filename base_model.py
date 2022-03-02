import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_model, get_optimizer, weight_init


class BaseModel:
    def __init__(self, model_type, device="cuda", save_folder="", num_cls=10,
                 optimizer="", lr=0.01, weight_decay=0, input_dim=100, epochs=0):
        self.model = get_model(model_type, num_cls, input_dim)
        self.model.to(device)
        self.model.apply(weight_init)
        self.device = device
        self.optimizer = get_optimizer(optimizer, self.model.parameters(), lr, weight_decay)
        if epochs == 0:
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[epochs // 2, epochs * 3 // 4], gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.save_pref = save_folder
        self.num_cls = num_cls

    def train(self, train_loader, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        if self.scheduler:
            self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def train_defend_ppb(self, train_loader, log_pref="",defend_arg=None):
        self.model.train()
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss1 = self.criterion(outputs, targets)
            ranked_outputs, _ = torch.topk(outputs, self.num_cls, dim=-1)
            size = targets.size(0)
            even_size = size // 2 * 2
            if even_size > 0:
                loss2 = F.kl_div(F.log_softmax(ranked_outputs[:even_size // 2], dim=-1),
                                 F.softmax(ranked_outputs[even_size // 2:even_size], dim=-1),
                                 reduction='batchmean')
            else:
                loss2 = torch.zeros(1).to(self.device)
            loss = loss1 + defend_arg * loss2
            total_loss += loss.item() * size
            total_loss1 += loss1.item() * size
            total_loss2 += loss2.item() * size
            total += size
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            loss.backward()
            self.optimizer.step()
        acc = 100. * correct / total
        total_loss /= total
        total_loss1 /= total
        total_loss2 /= total

        if self.scheduler:
            self.scheduler.step()
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}, Loss1 {:.3f}, Loss2 {:.3f}".format(
                log_pref, acc, total_loss, total_loss1, total_loss2))
        return acc, total_loss

    def test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                if isinstance(self.criterion, nn.BCELoss):
                    correct += torch.sum(torch.round(outputs) == targets)
                else:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def save(self, epoch, acc, loss):
        save_path = f"{self.save_pref}/{epoch}.pth"
        state = {
            'epoch': epoch + 1,
            'acc': acc,
            'loss': loss,
            'state': self.model.state_dict()
        }
        torch.save(state, save_path)
        return save_path

    def load(self, load_path, verbose=False):
        state = torch.load(load_path, map_location=self.device)
        acc = state['acc']
        if verbose:
            print(f"Load model from {load_path}")
            print(f"Epoch {state['epoch']}, Acc: {state['acc']:.3f}, Loss: {state['loss']:.3f}")
        self.model.load_state_dict(state['state'])
        return acc

    def predict_target_sensitivity(self, data_loader, m=10, epsilon=1e-3):
        self.model.eval()
        predict_list = []
        sensitivity_list = []
        target_list = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predicts = F.softmax(outputs, dim=-1)
                predict_list.append(predicts.detach().data.cpu())
                target_list.append(targets)

                if len(inputs.size()) == 4:
                    x = inputs.repeat((m, 1, 1, 1))
                elif len(inputs.size()) == 3:
                    x = inputs.repeat((m, 1, 1))
                elif len(inputs.size()) == 2:
                    x = inputs.repeat((m, 1))
                u = torch.randn_like(x)
                evaluation_points = x + epsilon * u
                new_predicts = F.softmax(self.model(evaluation_points), dim=-1)
                diff = torch.abs(new_predicts - predicts.repeat((m, 1)))
                diff = diff.view(m, -1, self.num_cls)
                sensitivity = diff.mean(dim=0) / epsilon
                sensitivity_list.append(sensitivity.detach().data.cpu())

        targets = torch.cat(target_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        sensitivities = torch.cat(sensitivity_list, dim=0)
        return predicts, targets, sensitivities
