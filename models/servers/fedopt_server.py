import copy
import torch
import torch.optim as optim
from collections import OrderedDict
from cifar100.retrain_model import ReTrainModel
from cifar100.dataset import VRDataset, MyImageDataset, MyTabularDataset
import numpy as np
from baseline_constants import conf
import torch.nn.functional as F

from .fedavg_server import Server


class FedOptServer(Server):
    def __init__(self, client_model, server_opt, server_lr, test_data, momentum=0, opt_ckpt=None):
        super().__init__(client_model)
        print("Server optimizer:", server_opt, "with lr", server_lr, "and momentum", momentum)
        self.server_lr = server_lr
        self.server_momentum = momentum
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=conf["batch_size"], shuffle=True)
        self.server_opt = self._get_optimizer(server_opt)
        if opt_ckpt is not None:
            self.load_optimizer_checkpoint(opt_ckpt)
       

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, analysis=False):
        self.server_opt.zero_grad()
        sys_metrics = super(FedOptServer, self).train_model(num_epochs, batch_size, minibatch, clients, analysis)
        self._save_updates_as_pseudogradients()
        return sys_metrics

    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """
        self.client_model.load_state_dict(self.model)
        # Average deltas and obtain global pseudo gradient (fedavg)
        pseudo_gradient = self._average_updates()
        # Update global model according to chosen optimizer
        self._update_global_model_gradient(pseudo_gradient)
        self.model = copy.deepcopy(self.client_model.state_dict())
        self.total_grad = self._get_model_total_grad()
        self.updates = []
        return

    def save_model(self, round, ckpt_path, swa_n=None):
        """Saves the servers model and optimizer on checkpoints/dataset/model.ckpt."""
        # Save servers model
        save_info = {'model_state_dict': self.model,
                     'opt_state_dict': self.server_opt.state_dict(),
                     'round': round}
        if self.swa_model is not None:
            save_info['swa_model'] = self.swa_model.state_dict()
        if swa_n is not None:
            save_info['swa_n'] = swa_n
        torch.save(save_info, ckpt_path)
        return ckpt_path

    def _save_updates_as_pseudogradients(self):
        clients_models = copy.deepcopy(self.updates)
        self.updates = []
        for i, (num_samples, update) in enumerate(clients_models):
            delta = self._compute_client_delta(update)
            self.updates.append((num_samples, delta))

    def _compute_client_delta(self, cmodel):
        """Args:
            cmodel: client update, i.e. state dict of client's update.
        Returns:
            delta: delta between client update and global model. """
        delta = OrderedDict.fromkeys(cmodel.keys()) # (delta x_i)^t
        for k, x, y in zip(self.model.keys(), self.model.values(), cmodel.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta

    def _update_global_model_gradient(self, pseudo_gradient):
        """Args:
            pseudo_gradient: global pseudo gradient, i.e. weighted average of the trained clients' deltas.

        Updates the global model gradient as -1.0 * pseudo_gradient
        """
        for n, p in self.client_model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]

        self.server_opt.step()

        bn_layers = OrderedDict(
            {k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.client_model.load_state_dict(bn_layers, strict=False)

    def _get_model_total_grad(self):
        """Returns:
            total_grad: sum of the L2-norm of the gradient of each trainable parameter"""
        total_norm = 0
        for name, p in self.client_model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except Exception:
                    # this param had no grad
                    pass
        total_grad = total_norm ** 0.5
        # print("total grad norm:", total_grad)
        return total_grad

    def _get_optimizer(self, server_opt):
        """Returns optimizer given its name. If not allowed, NotImplementedError expcetion is raised."""
        if server_opt == 'sgd':
            return optim.SGD(params=self.client_model.parameters(), lr=self.server_lr, momentum=self.server_momentum)
        elif server_opt == 'adam':
            return optim.Adam(params=self.client_model.parameters(), lr=self.server_lr, betas=(0.9, 0.99), eps=10**(-1))
        elif server_opt == 'adagrad':
            return optim.Adagrad(params=self.client_model.parameters(), lr=self.server_lr, eps=10**(-2))
        raise NotImplementedError

    def load_optimizer_checkpoint(self, optimizer_ckpt):
        """Load optimizer state from checkpoint"""
        self.server_opt.load_state_dict(optimizer_ckpt)
        
    #################### METHODS FOR FED-CCVR ####################

    @torch.no_grad()
    def model_eval_vr(self, eval_vr, label):
        """
        :param eval_vr:
        :param label:
        :return: 测试重训练模型
        """

        self.retrain_model.eval()

        eval_dataset = VRDataset(eval_vr, label)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=conf["batch_size"], shuffle=True)

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.functional.cross_entropy()
        for batch_id, batch in enumerate(eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.retrain_model(data)

            total_loss += criterion(output, target)  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss.cpu().detach().numpy() / dataset_size
        return acc, total_l

    def retrain_vr(self, vr, label, eval_vr, classifier):
        """
        :param vr:
        :param label:
        :return: the post-processed model.
        """
        self.retrain_model = classifier
        retrain_dataset = VRDataset(vr, label)
        retrain_loader = torch.utils.data.DataLoader(retrain_dataset, batch_size=conf["batch_size"],shuffle=True)

        optimizer = torch.optim.SGD(self.retrain_model.parameters(), lr=conf['retrain']['lr'], momentum=conf['momentum'],weight_decay=conf["weight_decay"])
        # optimizer = torch.optim.Adam(self.local_model.parameters(), lr=conf['lr'])
        criterion = torch.nn.CrossEntropyLoss()
        for e in range(conf["retrain"]["epoch"]):

            self.retrain_model.train()

            for batch_id, batch in enumerate(retrain_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.retrain_model(data)

                loss = criterion(output, target)
                loss.backward()

                optimizer.step()

            acc, eval_loss = self.model_eval_vr(eval_vr, label)
            print("Retraining epoch {0} done. train_loss ={1}, eval_loss = {2}, eval_acc={3}".format(e, loss, eval_loss, acc))

        return self.retrain_model

    def cal_global_gd(self,client_mean, client_cov, train_clients, test_clients):
        """
        :param client_mean: dictionary of mean values of features for each client
        :param client_cov:  dictionary of covariance matrices of features for each client
        :param client_length: dictionary of data count for each category and client (n_ck, #samples of class c on client k)
        :return: global mean and covariance matrices
        """

        g_mean = []
        g_cov = []

        clients = list(client_mean.keys())

        for c in range(1,conf["num_classes"]):

            mean_c = np.zeros_like(client_mean[clients[0]][0])
            # n_c is the total number of samples of class c for all the clients
            n_c = 0

            # total number of samples for class c
            for client in train_clients:
                print(client.num_samples_per_class[c])
                n_c += client.num_samples_per_class[c]

            cov_ck = np.zeros_like(client_cov[clients[0]][0])
            mul_mean = np.zeros_like(client_cov[clients[0]][0])

            for client in train_clients:

                # local mean
                mean_ck = np.array(client_mean[client.id][c])
                # global mean
                mean_c += (client.num_samples_per_class[c] / n_c) * mean_ck  # equation (3)


                cov_ck += ((client.num_samples_per_class[c] - 1) / (n_c - 1)) * np.array(client_cov[client.id][c]) # first term in equation (4)
                mul_mean += ((client.num_samples_per_class[c]) / (n_c - 1)) * np.dot(mean_ck.T, mean_ck) # second term in equation (4)


            g_mean.append(mean_c)

            # global covariance
            cov_c = cov_ck + mul_mean - (n_c / (n_c - 1)) * np.dot(mean_c.T, mean_c)  # equation (4)

            g_cov.append(cov_c)
            
            for client in test_clients:
                print(client.num_samples_per_class[c])
                n_c += client.num_samples_per_class[c]

            cov_ck = np.zeros_like(client_cov[clients[0]][0])
            mul_mean = np.zeros_like(client_cov[clients[0]][0])

            for client in test_clients:

                # local mean
                mean_ck = np.array(client_mean[client.id][c])
                # global mean
                mean_c += (client.num_samples_per_class[c] / n_c) * mean_ck  # equation (3)


                cov_ck += ((client.num_samples_per_class[c] - 1) / (n_c - 1)) * np.array(client_cov[client.id][c]) # first term in equation (4)
                mul_mean += ((client.num_samples_per_class[c]) / (n_c - 1)) * np.dot(mean_ck.T, mean_ck) # second term in equation (4)


            g_mean.append(mean_c)

            # global covariance
            cov_c = cov_ck + mul_mean - (n_c / (n_c - 1)) * np.dot(mean_c.T, mean_c)  # equation (4)

            g_cov.append(cov_c)

        return g_mean, g_cov


    def get_dataset(conf, data):
        """
        :param conf: 配置
        :param data: 数据 (DataFrame)
        :return:
        """
        if conf['data_type'] == 'tabular':
            dataset = MyTabularDataset(data, conf['label_column'])
        elif conf['data_type'] == 'image':
            dataset = MyImageDataset(data, conf['data_column'], conf['label_column'])
        else:
            return None
        return dataset

    def get_feature_label(self, test_clients):
        self.client_model.eval()

        cnt = 0
        features = []
        true_labels = []
        pred_labels = []

        for client in test_clients:
            if self.swa_model is None:
                client.model.load_state_dict(self.model)
            else:
                client.model.load_state_dict(self.swa_model.state_dict())
        
            for data in client.testloader:
                
                input_tensor, labels_tensor = data[0].to(self.device), data[1].to(self.device)
                with torch.no_grad():
                    outputs, feature = self.client_model(input_tensor)
                    _, pred = torch.max(outputs.data, 1)  # same as torch.argmax()
                    features.append(feature)
                    true_labels.append(labels_tensor)
                    pred_labels.append(pred)
                    cnt += input_tensor.size()[0]
                
                    if cnt > 1000:
                        break

        features = torch.cat(features, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)

        return features, true_labels, pred_labels