# Author: Guanxiong


import argparse
import yaml
import os
from datetime import datetime
from dataset import (
    PerMotionDataName,
    PerMotionDataset)
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from inertia_models import (
    UnconNetBase,
    SymmNetBase,
    SPDNetBase,
    UnconNetQVel,
    SymmNetQVel,
    SPDNetQVel,
    CRBA)
from inertia_losses import LossName, ImpulseLoss
from tqdm import tqdm
from Utils.inertia_utils import move_dict_to_device
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        '--config', type=str, help='path to yaml experiment/data config file', required=True)
    args = p.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # parse est/data config
    config_name = config["config_name"]
    h5_path = config["h5_path"]
    motion_name = os.path.splitext(os.path.basename(h5_path))[0].split("seq_name=")[1]
    time_curr = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    seq_length = config["seq_length"]
    estimation_specs = config["estimation_specs"]
    num_train_steps = estimation_specs["num_train_steps"]
    learning_rate = estimation_specs["learning_rate"]
    batch_size = estimation_specs["batch_size"]
    steps_til_val = estimation_specs["steps_til_val"]
    visual_specs = config["visual_specs"]
    plot_dir_base = visual_specs["plot_dir"]
    tb_dir_base = visual_specs["tb_dir"]
    dof = 46
    device = "cuda"
    loss_name = LossName.IMPULSE_LOSS
    
    # load data
    dataset = PerMotionDataset(h5_path, seq_length)
    train_size = int(len(dataset) * 0.8)
    test_size = int(len(dataset) * 0.1)
    val_size = len(dataset) - train_size - test_size
    dataset_train, dataset_val, dataset_test = random_split(
        dataset,
        [train_size, val_size, test_size])
    print(f"Loaded and splitted dataset.")
    print(f"Train size: {train_size}")
    print(f"Val size: {val_size}")
    print(f"Test size: {test_size}")

    # for each model specified, train + val + test
    for model_name, model_specs in estimation_specs["models"].items():
        # define inertia model
        # extract common NN specs
        network = model_specs["network"]
        if "activation" in model_specs:
            activation = model_specs["activation"]
        # decode MLP widths from depth
        if "mlp" in model_specs and "depth" in model_specs["mlp"]:
            mlp_depth = model_specs["mlp"]["depth"]
            if mlp_depth == 2:
                mlp_widths = [512]
            elif mlp_depth == 3:
                mlp_widths = [1024, 512]
            elif mlp_depth == 8:
                mlp_widths = [8192, 4096, 2048, 1024, 1024, 512, 512]
            elif mlp_depth == 12:
                mlp_widths = [4096] + \
                    [2048]*2 + \
                    [1024]*2 + \
                    [1024]*2 + \
                    [512]*2 + \
                    [512]*2
            elif mlp_depth == 16:
                mlp_widths = [16384] + \
                    [8192]*2 + \
                    [4096]*2 + \
                    [2048]*2 + \
                    [1024]*2 + \
                    [1024]*2 + \
                    [512]*2 + \
                    [512]*2
            else:
                raise ValueError("Unsupported MLP depth")
        # instantiate NN
        if network == "UnconNetBase":
            model = UnconNetBase(
                mlp_widths = mlp_widths,
                seq_length = seq_length,
                dof = dof,
                activation = activation).to(device)
        elif network == "SymmNetBase":
            model = SymmNetBase(
                mlp_widths = mlp_widths,
                seq_length = seq_length,
                dof = dof,
                activation = activation).to(device)
        elif network == "SPDNetBase":
            model = SPDNetBase(
                mlp_widths = mlp_widths,
                seq_length = seq_length,
                dof = dof,
                activation = activation,
                spd_layer_opts = model_specs["spd_layer"]).to(device)
        elif network == "UnconNetQVel":
            model = UnconNetQVel(
                mlp_widths = mlp_widths,
                seq_length = seq_length,
                dof = dof,
                activation = activation).to(device)
        elif network == "SymmNetQVel":
            model = SymmNetQVel(
                mlp_widths = mlp_widths,
                seq_length = seq_length,
                dof = dof,
                activation = activation).to(device)
        elif network == "SPDNetQVel":
            model = SPDNetQVel(
                mlp_widths = mlp_widths,
                seq_length = seq_length,
                dof = dof,
                activation = activation,
                spd_layer_opts = model_specs["spd_layer"]).to(device)
        elif network == "CRBA":
            model = CRBA()
        else:
            raise ValueError("Invalid network name")
        
        # define the loss class
        if loss_name == LossName.IMPULSE_LOSS:
            loss_cls = ImpulseLoss(dof=dof, reduction="mean")

        # define tb writer
        tb_dir_per_model = os.path.join(
            tb_dir_base,
            f"{config_name}+{motion_name}+t={time_curr}/",
            model_name)
        writer = SummaryWriter(log_dir = tb_dir_per_model)

        # create dirs for data I/O and visualization
        os.makedirs(tb_dir_per_model, exist_ok=True)

        # train iteratively
        print(f"Training {model_name}...")
        dataloader_train = DataLoader(
            dataset_train,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0)
        iterator_train = iter(dataloader_train)
        if network != "CRBA":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        for step in tqdm(range(num_train_steps)):
            # load batch
            try:
                data = next(iterator_train)
            except StopIteration:
                iterator_train = iter(dataloader_train)
                data = next(iterator_train)
            data_device = move_dict_to_device(
                data, device)
            model_input = {
                "qpos": data_device[PerMotionDataName.QPOS_GT],
                "qvel": data_device[PerMotionDataName.QVEL_GT],
                "M_rigid": data_device[PerMotionDataName.M_RIGID]}
            
            # forward pass
            if network != "CRBA":
                optimizer.zero_grad()
            model_output = model(model_input)

            # compute loss and backprop
            # qfrc_sum = \
            #     data_device[PerMotionDataName.QFRC_GR_OPT] + \
            #     data_device[PerMotionDataName.TAU_OPT] + \
            #     data_device[PerMotionDataName.GRAVCOL]
            qfrc_sum = data_device[PerMotionDataName.TAU_OPT]
            loss_dict = loss_cls.loss(
                model_output["inertia"],
                model_input["qvel"],
                qfrc_sum,
                torch.FloatTensor([0.011]).to(device))
            loss = loss_dict["loss"]
            if network != "CRBA":
                loss.backward()
                optimizer.step()
            writer.add_scalar("train/loss", loss, step)

            # validate
            # TODO:
