import abc
import logging
import os
import pickle
import random
from typing import List
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.autograd import Variable
from task_utils.training import build_optimizer
from collections import defaultdict
import time
import copy
from tqdm import tqdm
from task_utils.tools import dict2Obj
from task_utils.training import get_device
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from task_utils.init_distributed import initialize_distributed


class Algorithm(metaclass=abc.ABCMeta):

    def __init__(self, module_name, name, seed, out_dir, log_tensorboard):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        self.prediction_details = {}
        self.out_dir = out_dir
        self.log_tensorboard = log_tensorboard

        if self.out_dir:
            if log_tensorboard:
                self.writer = SummaryWriter(os.path.join(self.out_dir, "tensorboard_logs"))

        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, X):
        """
        Train the algorithm on the given dataset
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return anomaly score
        """

    def set_output_dir(self, out_dir):
        self.out_dir = out_dir

    def get_val_err(self):
        """
        :return: reconstruction error_tc for validation set,
        dimensions of num_val_time_points x num_channels
        Call after training
        """
        return None

    def get_val_loss(self):
        """
        :return: scalar loss after training
        """
        return None

    def write_loss(self, losses, epoch):
        for key, value in losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)


class PyTorchAlgorithm(Algorithm):

    def __init__(self, gpu, module_name, name, seed, out_dir, log_tensorboard, distributed=False, verbose=False, autocast=False):
        super().__init__(module_name, name, seed, out_dir, log_tensorboard)
        self.gpu = gpu
        self.seed = seed
        self.autocast = autocast
        self.distributed = distributed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = True
            #torch.backends.cudnn.deterministic = True
            # Set a fixed value for the hash seed
            os.environ["PYTHONHASHSEED"] = str(seed)

        self.torch_save = True
        self.additional_params = dict()
        self.additional_params["losses"] = defaultdict(list)
        self.verbose = verbose

        if self.distributed:
            self.local_rank = initialize_distributed(server="bsi")
            self.gpu = torch.device("cuda:" + str(self.local_rank))

    @property
    def device(self):
        return self.gpu if self.gpu is not None else 'cpu'

    def to_var(self, t, **kwargs):
        # ToDo: check whether cuda Variable.
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, model):
        model.to(self.device)

    def to_tensor_cuda(self, ts_batch, device):
        if isinstance(ts_batch, list):
            return [item.float().to(device) for item in ts_batch]
        else:
            return ts_batch.float().to(device)

    def train_step(self, ts_batch):
        raise NotImplementedError

    def val_step(self, ts_batch):
        raise NotImplementedError

    def fit_wo_early_stopping(self, train_loader):

        self.model.to(self.device)

        scheduler, optimizer = build_optimizer(dict2Obj(self.init_params), self.model.parameters())

        self.model.train()

        if self.autocast:
            scaler = torch.cuda.amp.GradScaler()

        # assuming first batch is complete
        train_start = time.time()

        for epoch in range(self.num_epochs):
            logging.debug(f'Epoch {epoch + 1}/{self.num_epochs}.')
            self.model.train()
            train_loss = []
            loss_dict_per_epoch = defaultdict(list)
            for ts_batch in tqdm(train_loader):
                #with torch.autograd.detect_anomaly():
                self.model.zero_grad()

                ts_batch = self.to_tensor_cuda(ts_batch, self.device)

                if self.autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = self.train_step(ts_batch)
                else:
                    loss = self.train_step(ts_batch)

                if isinstance(loss, tuple):
                    loss, loss_dict_ = loss
                    for key, value in loss_dict_.items():
                        loss_dict_per_epoch[key].append(value.item())

                if self.autocast:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                # multiplying by length of batch to correct accounting for incomplete batches
                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)
            self.additional_params["losses"]["train_loss_by_epoch"].append(train_loss)
            for key, value in loss_dict_per_epoch.items():
                self.additional_params["losses"][f"train_{key}"].append(np.mean(value))
            if self.verbose:
                print(f"epoch: {epoch},", {name: value[-1] for name, value in self.additional_params["losses"].items()})
                if self.out_dir:
                    self.write_loss(self.additional_params["losses"], epoch)
            if scheduler is not None:
                scheduler.step()

        train_time = int(time.time() - train_start)
        if self.out_dir is not None:
            if self.log_tensorboard:
                self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def fit_with_early_stopping(self, train_loader, val_loader):

        self.model.to(self.device)#.double()

        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        #self.model = torch.compile(self.model)
        scheduler, optimizer = build_optimizer(dict2Obj(self.init_params), self.model.parameters())
        epoch_wo_improv = 0
        self.model.train()
        best_val_loss = None
        best_params = self.model.state_dict()

        # assuming first batch is complete
        train_start = time.time()


        if self.autocast:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.num_epochs):
            if epoch_wo_improv < self.patience:
                logging.debug(f'Epoch {epoch + 1}/{self.num_epochs}.')
                self.model.train()
                train_loss = torch.Tensor([0]).to(self.device)
                num_sample = torch.Tensor([0]).to(self.device)
                pbar = self.get_tqdm(train_loader)
                loss_dict_per_epoch_cum = defaultdict(list)
                for i, ts_batch in enumerate(pbar):
                    self.model.zero_grad()
                    ts_batch = self.to_tensor_cuda(ts_batch, self.device)
                    if self.autocast:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            loss = self.train_step(ts_batch)
                    else:
                        loss = self.train_step(ts_batch)
                    if isinstance(loss, tuple):
                        loss, loss_dict_ = loss
                        for key, value in loss_dict_.items():
                            if key in loss_dict_per_epoch_cum:
                                loss_dict_per_epoch_cum[key] += value
                            else:
                                loss_dict_per_epoch_cum[key] = value
                    if self.autocast:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    train_loss += loss
                    num_sample += 1

                    if (self.distributed and self.local_rank == 0) or not self.distributed:
                        if self.verbose:
                            pbar.set_postfix({"epoch": epoch, "batch iter": i, "train_loss": loss.item()})

                if self.distributed:
                    dist.barrier()
                    dist.all_reduce(train_loss)
                    dist.all_reduce(num_sample)
                    self.additional_params["losses"]["train_loss_by_epoch"].append((train_loss / num_sample).item())
                    for key, value in loss_dict_per_epoch_cum.items():
                        dist.all_reduce(value)
                        dist.all_reduce(num_sample)
                        self.additional_params["losses"][f"train_{key}"].append((value / num_sample).item())
                else:
                    train_loss = train_loss / num_sample
                    self.additional_params["losses"]["train_loss_by_epoch"].append(train_loss.item())
                    for key, value in loss_dict_per_epoch_cum.items():
                        value = value / num_sample
                        self.additional_params["losses"][f"train_{key}"].append(value.item())

                # Get Validation loss
                self.model.eval()
                val_loss_cum = torch.Tensor([0]).to(self.device)
                num_sample = torch.Tensor([0]).to(self.device)
                loss_dict_per_epoch_cum = {}
                with torch.no_grad():
                    for ts_batch in val_loader:
                        ts_batch = self.to_tensor_cuda(ts_batch, self.device)
                        if self.autocast:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                loss = self.val_step(ts_batch)
                        else:
                            loss = self.val_step(ts_batch)
                        if isinstance(loss, tuple):
                            loss, loss_dict_ = loss
                            for key, value in loss_dict_.items():
                                if key in loss_dict_per_epoch_cum:
                                    loss_dict_per_epoch_cum[key] += value
                                else:
                                    loss_dict_per_epoch_cum[key] = value
                        val_loss_cum += loss
                        num_sample += 1

                if self.distributed:
                    # dist.barrier()
                    dist.all_reduce(val_loss_cum)
                    dist.all_reduce(num_sample)
                    self.additional_params["losses"]["val_loss_by_epoch"].append((val_loss_cum / num_sample).item())

                    for key, value in loss_dict_per_epoch_cum.items():
                        dist.all_reduce(value)
                        dist.all_reduce(num_sample)
                        self.additional_params["losses"][f"val_{key}"].append((value / num_sample).item())
                else:
                    self.additional_params["losses"]["val_loss_by_epoch"].append((val_loss_cum / num_sample).item())
                    for key, value in loss_dict_per_epoch_cum.items():
                        self.additional_params["losses"][f"val_{key}"].append((value / num_sample).item())

                if (self.distributed and self.local_rank == 0) or not self.distributed:
                    print(f"epoch: {epoch},", {name: value[-1] for name, value in self.additional_params["losses"].items()})
                    if self.out_dir:
                        self.write_loss(self.additional_params["losses"], epoch)

                best_val_loss_epoch = np.argmin(self.additional_params["losses"]["val_loss_by_epoch"])
                if best_val_loss_epoch == epoch:
                    # any time a new best is encountered, the best_params will get replaced
                    best_params = copy.deepcopy(self.model.state_dict())
                    best_val_loss = self.additional_params["losses"]["val_loss_by_epoch"][-1]
                # Check for early stopping by counting the number of epochs since val loss improved
                if epoch > 0 and self.additional_params["losses"]["val_loss_by_epoch"][-1] > best_val_loss:
                    epoch_wo_improv += 1
                else:
                    epoch_wo_improv = 0
            else:
                # early stopping is applied
                self.model.load_state_dict(best_params)
                break
            if scheduler is not None:
                scheduler.step(epoch)

        train_time = time.time() - train_start

        if (self.distributed and self.local_rank == 0) or not self.distributed:
            print(f"-- Training done in {train_time}s.")

            if self.out_dir:
                if self.log_tensorboard:
                    self.writer.add_text("total_train_time", str(train_time))

    def get_tqdm(self, iterable):

        if self.verbose:
            if self.distributed:
                if self.local_rank == 0:
                    return tqdm(iterable, ncols=150, position=0, leave=True)
                else:
                    return iterable
            return tqdm(iterable, ncols=150, position=0, leave=True)
        else:
            return iterable

def load_torch_algo(algo_class, out_dir, eval=True, device=None):
    """
    :param algo_class: Class of the Algorithm to be instantiated
    :param out_dir: path to the directory where everything is to be saved
    :param eval: boolean to determine if model is to be put in evaluation mode
    :return: object of algo_class with a trained model
    """

    algo_config_filename = os.path.join(out_dir, "init_params")
    saved_model_filename = os.path.join(out_dir, "trained_model")
    additional_params_filename = os.path.join(out_dir, "additional_params")

    with open(os.path.join(algo_config_filename), "rb") as file:
        init_params = pickle.load(file)

    with open(additional_params_filename, "rb") as file:
        additional_params = pickle.load(file)

    # init params must contain only arguments of algo_class's constructor
    algo = algo_class(**init_params)
    if device is not None:
        algo.gpu = device
    else:
        device = get_device()#algo.device
        algo.gpu = device
    if additional_params is not None:
        setattr(algo, "additional_params", additional_params)

    if isinstance(saved_model_filename, List):
        algo.model = [torch.load(path, map_location=device) for path in saved_model_filename]
        if eval:
            [model.eval() for model in algo.model]
    else:
        algo.model = torch.load(saved_model_filename, map_location=device)#.to(device)
        algo.model.gpu = device
        if eval:
            algo.model.eval()
    return algo


def save_torch_algo(algo: Algorithm, out_dir):
    """
    Save the trained model and the hyper parameters of the algorithm
    :param algo: the algorithm object
    :param out_dir: path to the directory where everything is to be saved
    :return: Nothing
    """
    if isinstance(algo.model, List):
        saved_model_filename = []
        for k in range(len(algo.model)):
            model_filename = os.path.join(out_dir, "trained_model_channel_%i" % k)
            saved_model_filename.append(model_filename)
            torch.save(algo.model[k], model_filename)
    else:
        saved_model_filename = os.path.join(out_dir, "trained_model")
        torch.save(algo.model, saved_model_filename)
    init_params = algo.init_params
    algo_config_filename = os.path.join(out_dir, "init_params")
    with open(algo_config_filename, "wb") as file:
        pickle.dump(init_params, file)

    additional_params_filename = os.path.join(out_dir, "additional_params")
    additional_params = algo.additional_params
    with open(additional_params_filename, "wb") as file:
        pickle.dump(additional_params, file)

    return saved_model_filename, algo_config_filename, additional_params_filename