"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim


import torchvision.models as models
from src.dataloader import create_dataloader
# from src.loss import CustomCriterion
from src.loss import create_criterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info
from src.scheduler import CosineAnnealingWarmUpRestarts

from adamp import SGDP

def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, 'model.yml'), 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model=models.shufflenet_v2_x0_5(pretrained=False)
    model.conv5[0].out_channels=224
    model.conv5[1].num_features=224
    model.fc.in_features=224
    model.fc.out_features=9
    print(model)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")


    optimizer = SGDP(model.parameters(), lr=0.000001, weight_decay=1e-5, momentum=0.9, nesterov=True)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=0.03,  T_up=5, gamma=0.1)

    criterion = create_criterion('cross_entropy')
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", default="/home/kyungmin/pstage4/code/configs/model/shufflenet.yaml", type=str, help="model config"
    )
    parser.add_argument(
        "--data", default="configs/data/taco1.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("exp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
