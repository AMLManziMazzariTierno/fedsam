import copy
import numpy as np
import os
import random
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, CLIENT_PARAMS_KEY, CLIENT_GRAD_KEY, CLIENT_TASK_KEY, conf
from collections import OrderedDict
from tqdm import tqdm
from cifar100.retrain_model import ReTrainModel
from cifar100.dataset import VRDataset, MyImageDataset, MyTabularDataset


class Server:

    def __init__(self, client_model):
        self.client_model = copy.deepcopy(client_model)
        self.device = self.client_model.device
        self.model = copy.deepcopy(client_model.state_dict())
        self.total_grad = 0
        self.selected_clients = []
        self.updates = []
        self.momentum = 0
        self.swa_model = None

    #################### METHODS FOR FEDERATED ALGORITHM ####################

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the servers can select.
            num_clients: Number of clients to select; default 20.
            my_round: Current round.
        Return:
            list of (num_train_samples, num_test_samples)
        """

        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, analysis=False):
        """Trains self.model on given clients.

        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to servers
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from servers
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   CLIENT_PARAMS_KEY: 0,
                   CLIENT_GRAD_KEY: 0,
                   CLIENT_TASK_KEY: {}
                   } for c in clients}

        for c in clients:
            c.model.load_state_dict(self.model)
            num_samples, update = c.train(num_epochs, batch_size, minibatch)
            sys_metrics = self._update_sys_metrics(c, sys_metrics)
            self.updates.append((num_samples, copy.deepcopy(update)))

        return sys_metrics

    def _update_sys_metrics(self, c, sys_metrics):
        if isinstance(c.model, torch.nn.DataParallel):
            sys_metrics[c.id][BYTES_READ_KEY] += c.model.module.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.module.size
        else:
            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size

        sys_metrics[c.id][CLIENT_PARAMS_KEY] = c.params_norm()
        sys_metrics[c.id][CLIENT_GRAD_KEY] = c.total_grad_norm()
        sys_metrics[c.id][CLIENT_TASK_KEY] = c.get_task_info()
        return sys_metrics

    def test_model(self, clients_to_test, batch_size, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.
        For each client, the current servers model is loaded before testing it.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            if self.swa_model is None:
                client.model.load_state_dict(self.model)
            else:
                client.model.load_state_dict(self.swa_model.state_dict())
            c_metrics = client.test(batch_size, set_to_use)
            metrics[client.id] = c_metrics

        return metrics

    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """
        # Average updates (fedavg)
        averaged_soln = self._average_updates()
        # Update global model
        self.client_model.load_state_dict(averaged_soln)
        self.model = copy.deepcopy(self.client_model.state_dict())
        self.updates = []
        return

    def _average_updates(self):
        """ Method for performing FedAvg: Weighted average of self.updates, where the weight is given by the number
        of samples seen by the corresponding client at training time. """
        total_weight = 0.
        base = OrderedDict()
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += (client_samples * value.type(torch.FloatTensor))
                else:
                    base[key] = (client_samples * value.type(torch.FloatTensor))

        averaged_soln = copy.deepcopy(self.model)
        for key, value in base.items():
            if total_weight != 0:
                averaged_soln[key] = value.to(self.device) / total_weight
        return averaged_soln

    def update_clients_lr(self, lr, clients=None):
        if clients is None:
            clients = self.selected_clients
        for c in clients:
            c.update_lr(lr)

    #################### SWA METHODS ####################

    def setup_swa_model(self, swa_ckpt=None):
        self.swa_model = copy.deepcopy(self.client_model)
        if swa_ckpt is not None:
            self.swa_model.load_state_dict(swa_ckpt)

    def update_swa_model(self, alpha):
        for param1, param2 in zip(self.swa_model.parameters(), self.client_model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha

    #################### METHODS FOR ANALYSIS ####################

    def set_num_clients(self, n):
        """Sets the number of total clients"""
        self.num_clients = n

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.
        Returns info about self.selected_clients if clients=None;
        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, num_samples

    def num_parameters(self, params):
        """Number of model parameters requiring training"""
        return sum(p.numel() for p in params if p.requires_grad)

    def get_model_params_norm(self):
        """Returns:
            total_params_norm: L2-norm of the model parameters"""
        total_norm = 0
        for p in self.client_model.parameters():
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_params_norm = total_norm ** 0.5
        return total_params_norm

    def get_model_grad(self):
        """Returns:
            self.total_grad: total gradient of the model (zero in case of FedAvg, where the gradient is never stored)"""
        return self.total_grad

    def get_model_grad_by_param(self):
        """Returns:
            params_grad: dictionary containing the L2-norm of the gradient for each trainable parameter of the network
                        (zero in case of FedAvg where the gradient is never stored)"""
        params_grad = {}
        for name, p in self.client_model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(2)
                    params_grad[name] = param_norm
                except Exception:
                    # this param had no grad
                    pass
        return params_grad

    #################### METHODS FOR SAVING CHECKPOINTS ####################

    def save_model(self, round, ckpt_path, swa_n=None):
        """Saves the servers model on checkpoints/dataset/model_name.ckpt."""
        # Save servers model
        save_info = {'model_state_dict': self.model,
                    'round': round}
        if self.swa_model is not None:
            save_info['swa_model'] = self.swa_model.state_dict()
        if swa_n is not None:
            save_info['swa_n'] = swa_n
        torch.save(save_info, ckpt_path)
        return ckpt_path

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

    def cal_global_gd(self,client_mean, client_cov, client_length):
        """
        :param client_mean: dictionary of mean values of features for each client
        :param client_cov:  dictionary of covariance matrices of features for each client
        :param client_length: dictionary of data count for each category and client (n_ck, #samples of class c on client k)
        :return: global mean and covariance matrices
        """

        g_mean = []
        g_cov = []

        clients = list(client_mean.keys())

        for c in range(len(client_mean[clients[0]])):

            mean_c = np.zeros_like(client_mean[clients[0]][0])
            # n_c is the total number of samples of class c for all the clients
            n_c = 0

            # total number of samples for class c
            for k in clients:
                n_c += client_length[k][c]

            cov_ck = np.zeros_like(client_cov[clients[0]][0])
            mul_mean = np.zeros_like(client_cov[clients[0]][0])

            for k in clients:

                # local mean
                mean_ck = np.array(client_mean[k][c])
                # global mean
                mean_c += (client_length[k][c] / n_c) * mean_ck  # equation (3)


                cov_ck += ((client_length[k][c] - 1) / (n_c - 1)) * np.array(client_cov[k][c]) # first term in equation (4)
                mul_mean += ((client_length[k][c]) / (n_c - 1)) * np.dot(mean_ck.T, mean_ck) # second term in equation (4)


            g_mean.append(mean_c)

            # global covariance
            cov_c = cov_ck + mul_mean - (n_c / (n_c - 1)) * np.dot(mean_c.T, mean_c)  # equation (4)

            g_cov.append(cov_c)

        return g_mean, g_cov

def get_dataset(conf, data):
    if conf['data_type'] == 'tabular':
        dataset = MyTabularDataset(data, conf['label_column'])
    elif conf['data_type'] == 'image':
        dataset = MyImageDataset(data, conf['data_column'], conf['label_column'])
    else:
        return None
    return dataset

def get_feature_label(self):
    self.global_model.eval()
    
    cnt = 0
    features = []
    true_labels = []
    pred_labels = []
    for batch_id, batch in enumerate(self.test_loader):
        data, target = batch
        cnt += data.size()[0]

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        feature, output = self.global_model(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        
        features.append(feature)
        true_labels.append(target)
        pred_labels.append(pred)
        
        if cnt > 1000:
            break

    features = torch.cat(features, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)

    return features, true_labels, pred_labels