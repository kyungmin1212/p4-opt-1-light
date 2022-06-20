"""Tune Model.

- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""
import argparse
import copy
import optuna
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info
from src.utils.common import read_yaml
from src.utils.macs import calc_macs
from src.trainer import TorchTrainer
from typing import Any, Dict, List, Tuple, Union
from train3 import train
import wandb
import matplotlib.pyplot as plt

MODEL_CONFIG = read_yaml(cfg="/home/kyungmin/pstage4/code/exp/2021-06-11_04-19-13/model.yml")
DATA_CONFIG = read_yaml(cfg="configs/data/taco1.yaml")

def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = trial.suggest_int("epochs", low=50, high=50, step=10)
    img_size = trial.suggest_categorical("img_size", [56])
    n_select = trial.suggest_int("n_select", low=0, high=6, step=1)
    level=trial.suggest_int("level",low=0,high=30,step=2)
    batch_size = trial.suggest_int("batch_size", low=64, high=64, step=16)
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "level":level,
        "BATCH_SIZE": batch_size
    }

def objective(trial: optuna.trial.Trial, device) -> Tuple[float, int, float]:
    global count
    """Optuna objective.

    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    wandb.init(project='search_nlevel_56',reinit=True)
    model_root=trial.suggest_categorical("model_root", ["exp/2021-06-11_04-19-13/model.yml","exp/2021-06-11_04-34-38/model.yml","exp/2021-06-11_04-03-59/model.yml","exp/2021-06-11_05-40-35/model.yml","exp/2021-06-11_13-41-17/model.yml"])
    MODEL_CONFIG = read_yaml(cfg=model_root)
    model_config = copy.deepcopy(MODEL_CONFIG)
    data_config = copy.deepcopy(DATA_CONFIG)
    wandb.run.name = f'{trial.number} trial'
    wandb.run.save()
    # hyperparams: EPOCHS, IMG_SIZE, n_select, BATCH_SIZE
    hyperparams = search_hyperparam(trial)
    model_config["input_size"] = [hyperparams["IMG_SIZE"], hyperparams["IMG_SIZE"]]
    # model_config["backbone"] = 

    data_config["AUG_TRAIN_PARAMS"]["n_select"] = hyperparams["n_select"]
    data_config["AUG_TRAIN_PARAMS"]["level"] = hyperparams["level"]
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["EPOCHS"] = hyperparams["EPOCHS"]
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]

    log_dir = os.path.join("exp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    wandb.config.update(hyperparams)
    wandb.config.update({'log_dir':log_dir,"model_root":model_root})
    os.makedirs(log_dir, exist_ok=True)

    # Calculate macs
    model_instance = Model(model_config, verbose=False)
    wandb.watch(model_instance.model)
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))

    # model_config, data_config
    _, test_f1, _ = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
    data=[[test_f1,macs]]
    table=wandb.Table(data=data,columns =["test_f1","macs"])
    wandb.log({"test_f1 and macs":wandb.plot.scatter(table,"test_f1","macs")})


    return test_f1


def tune(gpu_id: int, storage: Union[str, None] = None, study_name: str = "pstage_automl"):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    study = optuna.create_study(
        directions=["maximize"],
        storage=rdb_storage,
        study_name=study_name,
        sampler=sampler,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, device), n_trials=300)
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    # best_trial = get_best_trial_with_condition(study)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="postgresql://optuna:lkm961296@101.101.210.70:6013/optuna", type=str, help="RDB Storage URL for optuna.")
    parser.add_argument("--study-name", default="search_nlevel_56", type=str, help="Optuna study name.")
    args = parser.parse_args()
    tune(args.gpu, storage=None if args.storage == "" else args.storage, study_name=args.study_name)
