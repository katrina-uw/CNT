from base_task import PyTorchAlgorithm, save_torch_algo, load_torch_algo
from task_utils.datasets import get_train_data_loaders
import pandas as pd
import numpy as np
from task_utils.datasets import TsReconstructionDataset
import torch
from torch.utils.data import DataLoader
from model_utils.main_model import MainModel
from tqdm import tqdm

import time
import random
from task_utils.tools import dict2Obj


class CNT(PyTorchAlgorithm):
    def __init__(self, name: str='CNT', target_dims=None, seed: int=None, gpu: int = None, verbose=True, out_dir=None, log_tensorboard=False, **args):
        """
        Args to initialize MyModel:
        """
        PyTorchAlgorithm.__init__(self, gpu, __name__, name, seed=seed, out_dir=out_dir, log_tensorboard=log_tensorboard, verbose=verbose)
        self.__dict__.update(args)
        np.random.seed(seed)
        random.seed(seed)
        self.target_dims = target_dims

        self.init_params = {"name": name,
                            "seed": seed,
                            "out_dir": out_dir,
                            "gpu": gpu,
                            "target_dims": target_dims,
                            }
        self.init_params.update(args)

    def fit(self, X: pd.DataFrame, train_starts=np.array([])):

        X.interpolate(inplace=True)
        X.bfill(inplace=True)

        data = X.values

        train_data = TsReconstructionDataset(data, self.seq_len, self.stride, starts=train_starts)

        self.model = MainModel(dict2Obj(self.init_params))

        train_loader, train_val_loader = get_train_data_loaders(train_data, batch_size=self.batch_size, \
                                                                train_val_percentage=self.train_val_percentage, seed=self.seed, shuffle=True, num_workers=self.num_workers)

        self.fit_with_early_stopping(train_loader, train_val_loader)

    def to_tensor_cuda(self, ts_batch, device):
        if isinstance(ts_batch, list):
            result =  [item.float().to(device) for item in ts_batch]
        else:
            result = ts_batch.to(device)
        return result

    def train_step(self, ts_batch):

        model_input = ts_batch.float()
        output = self.model(model_input)
        loss_dict = self.model.get_loss(*output, reduction=True)
        return sum(loss_dict.values()), loss_dict

    def val_step(self, ts_batch):

        model_input = ts_batch.float()
        output = self.model(model_input)
        loss_dict = self.model.get_loss(*output, reduction=True)
        return sum(loss_dict.values()), loss_dict

    @torch.no_grad()
    def predict(self, X: pd.DataFrame, starts=np.array([])) -> np.array:
        test_start_time = time.time()
        X.interpolate(inplace=True)
        X.bfill(inplace=True)

        data = X.values

        test_data = TsReconstructionDataset(data, self.seq_len, self.stride, starts=np.array([]))
        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False, num_workers=self.num_workers)

        predictions_dic = self.predict_test_scores(test_loader)
        test_end_time = time.time()
        if self.verbose:
            print(f"Test time: {test_end_time - test_start_time}")
        return predictions_dic

    def get_error(self, output, ts_batch):

        contrast_errors = self.model.get_loss(*output, reduction=False)
        return contrast_errors

    @torch.no_grad()
    def predict_test_scores(self, test_loader):
        self.model.eval()
        error_dict = {"OCC": [], "NTL": [], "Graph": [], "infoNCE": []}
        error_score_dict = {"OCC": [], "NTL": [], "Graph": [], "infoNCE": []}
        predictions_dic = {}

        for index, ts_batch in enumerate(tqdm(test_loader)):
            ts_batch = self.to_tensor_cuda(ts_batch, self.device)
            model_input = ts_batch.float()
            output = self.model(model_input)
            errors = self.get_error(output, ts_batch)
            for error_type in errors:
                error_dict[error_type].append(errors[error_type].cpu().numpy())

        for error_type, error in error_dict.items():
            if len(error):
                error = np.concatenate(error)
                if len(error.shape) >= 2:
                    pre_padding = np.zeros((self.seq_len-self.win_size, error.shape[1]))
                    post_padding = np.zeros((self.win_size-1, error.shape[1]))
                else:
                    pre_padding = np.zeros((self.seq_len-self.win_size))
                    post_padding = np.zeros((self.win_size-1))
                error = np.concatenate([pre_padding, error, post_padding], axis=0)
                error_score_dict[error_type] = error
                if len(error.shape) >= 2:
                    predictions_dic[f"{error_type.lower()}_error_tc"] = error
                    predictions_dic[f"{error_type.lower()}_error_t"] = error.mean(axis=-1)
                else:
                    predictions_dic[f"{error_type.lower()}_error_t"] = error

        if self.error_type == "all":
            #predictions_dic["score_t"] = 1 / np.sum(np.vstack([(1 / (predictions_dic[t]+1e-5)) for t in predictions_dic if (t.endswith("_t") and t!= "score_t")]).T, axis=-1)
            predictions_dic["score_t"] = np.sum(np.vstack([predictions_dic[t] for t in predictions_dic if (t.endswith("_t") and t!= "score_t")]).T, axis=-1)
        elif self.error_type == "ntl":
            predictions_dic["score_t"] = predictions_dic["ntl_error_t"]
        elif self.error_type == "occ":
            predictions_dic["score_t"] = predictions_dic["occ_error_t"]
        elif self.error_type == "infoNCE":
            predictions_dic["score_t"] = predictions_dic["infoNCE_error_t"]
        return predictions_dic


