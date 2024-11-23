import argparse
from task_utils.training import get_device
import os
from base_task import save_torch_algo, load_torch_algo
from configs import anomaly_zscore_factor_dict, anomaly_percentile_dict, anomaly_ratio_dict
import torch
import pandas as pd
from task_utils.init_distributed import initialize_distributed
import torch.distributed as dist
from data_utils.dataset import get_dataset
from task_utils.evaluation.evaluator import Evaluator
from configs import reg_levels, get_best_config
from CNT_task import CNT as module


def train_and_test(dataset, seed, gpu, save_model, train=True, metric_types=["f1_pa", "f1", "f1_cp", "vus_pr", "vus_roc"]):


    if dataset.startswith("smd"):
        dataset_, entity = dataset.split("_")
        args = get_best_config("CNT", dataset_)
        ds = get_dataset(dataset_, normalize=args["normalize"], entity=entity, normalize_type=args["normalize_type"], seed=seed)
    elif dataset.startswith("hai"):
        dataset_, entity = dataset.split("_")
        args = get_best_config("CNT", dataset_)
        ds = get_dataset(dataset_, normalize=args["normalize"], entity=entity, normalize_type=args["normalize_type"], seed=seed)
    else:
        args = get_best_config("CNT", dataset)
        ds = get_dataset(dataset, normalize=args["normalize"], normalize_type=args["normalize_type"], seed=seed)


    x_train, y_train, x_test, y_test = ds.data()

    args["distributed"] = distributed
    if (distributed and local_rank == 0) or (not distributed):
        print(args)

    base_dir = os.path.join(current_dir, "outputs", "CNT", f"{dataset}", str(seed))
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    algo = module(**args, seed=seed, gpu=gpu, verbose=True, target_dims=ds.get_target_dims(), log_tensorboard=True, out_dir=base_dir)

    if train:
        algo.fit(x_train, train_starts=ds.get_start_position(is_train=True))
        if save_model:
            save_torch_algo(algo, out_dir=base_dir)
            algo = load_torch_algo(algo_class=module, out_dir=base_dir, device="cpu")
            if not distributed:
                algo.gpu = gpu
                algo.model.to(gpu)
            else:
                algo.gpu = torch.device("cuda:" + str(local_rank))
                algo.model.to(local_rank)
    else:
        algo = load_torch_algo(algo_class=module, out_dir=base_dir, device="cpu")
        if not distributed:
            algo.gpu = gpu
            algo.model.to(gpu)
        else:
            algo.gpu = torch.device("cuda:" + str(local_rank))
            algo.model.to(local_rank)
    if (distributed and local_rank == 0) or (not distributed):
        test_prediction = algo.predict(x_test, starts=ds.get_start_position(is_train=False))
        train_prediction = algo.predict(x_train, starts=ds.get_start_position(is_train=True))

        dataset = dataset.split("_")[0]

        evaluator = Evaluator(ds_object=ds, batch_size=args["batch_size"], reg_level=reg_levels[dataset], \
                              scale_scores=False, anomaly_percentile=anomaly_percentile_dict[dataset],
                              anomaly_ratio=anomaly_ratio_dict[dataset], \
                              anomaly_zscore_factor=anomaly_zscore_factor_dict[dataset])
        results = evaluator.evaluate(test_prediction, train_prediction, labels=y_test.values, metric_types=metric_types, save_path=None)

        return results
    else:
        return None


def get_results(datasets, seeds, gpu, save_model=False, train=True, metric_types=["f1_pa", "f1", "f1_cp", "vus_pr", "vus_roc"]):

    X = []
    for seed in seeds:
        for dataset in datasets:

            results = train_and_test(dataset, seed, gpu, save_model, train=train, metric_types=metric_types)

            if (distributed and local_rank == 0) or (not distributed):
                tmp_X = {"dataset": dataset, "seed": seed}
                tmp_X.update(results)
                X.append(tmp_X)
            if distributed:
                dist.barrier()

            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()

    if (distributed and local_rank == 0) or (not distributed):
        X = pd.DataFrame(X)
        return X
    else:
        return None


# main
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default="123,124,125,126,127")
    parser.add_argument('--datasets', type=str, default="swat,wadi,msl,smap,hai_1,hai_2,hai_3")
    parser.add_argument('--train', action="store_true", default=False)
    parser.add_argument('--distributed', action='store_true', default=False)

    current_dir = os.path.dirname(__file__)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    datasets = args.datasets.split(",")
    distributed = args.distributed
    if distributed:
        local_rank = initialize_distributed(server="bsi")

    metric_types = ["f1_pa", "f1", "f1_cp", "vus_pr", "vus_roc"]

    if distributed:
        gpu = None
    else:
        gpu = get_device()

    X = get_results(datasets=datasets, seeds=seeds, gpu=gpu, save_model=False, train=args.train, metric_types=metric_types)
    if (distributed and local_rank == 0) or (not distributed):
       print(X.groupby(["dataset", "seed"])[metric_types].mean())