# Author: Guanxiong


import argparse
import yaml
import os
from datetime import datetime
from dataset import (
    PerMotionDataName,
    PerMotionDataset)
import torch
import numpy as np
from torch.utils.data import (
    random_split,
    DataLoader,
    Subset)
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
from Visualizations.inertia_visualization import InertiaMatrixVisualizer


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
    force_specs = config["force_specs"]
    assert len(force_specs) > 0
    estimation_specs = config["estimation_specs"]
    num_train_steps = estimation_specs["num_train_steps"]
    optimizer_specs = estimation_specs["optimizer"]
    batch_size = estimation_specs["batch_size"]
    steps_til_val = estimation_specs["steps_til_val"]
    visual_specs = config["visual_specs"]
    plot_dir_base = visual_specs["plot_dir"]
    plot_dir_exp = os.path.join(
        plot_dir_base,
        f"{config_name}+{motion_name}+t={time_curr}/")
    tb_dir_base = visual_specs["tb_dir"]
    tb_dir_exp = os.path.join(
        tb_dir_base,
        f"{config_name}+{motion_name}+t={time_curr}/")
    dof = 46
    n_iters = 6
    dt_dcycle = 0.011
    dt = n_iters * dt_dcycle
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

    # randomly pick three test sequences for visual
    test_seq_idxs = torch.randperm(len(dataset_test))[:3]
    dataloader_test = DataLoader(
        Subset(dataset_test, test_seq_idxs),
        batch_size = len(test_seq_idxs),
        shuffle = False,
        num_workers = 0)
    iterator_test = iter(dataloader_test)
    
    # for each model specified, train + val + test
    test_results = {}
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
            model = CRBA(
                model_specs["urdf_path"])
        else:
            raise ValueError("Invalid network name")
        
        # define the loss class
        if loss_name == LossName.IMPULSE_LOSS:
            loss_cls = ImpulseLoss(dof=dof, reduction="mean")

        # define optimizer
        if network != "CRBA":
            if optimizer_specs["name"] == "Adam":
                learning_rate = optimizer_specs["params"]["learning_rate"]
                amsgrad = optimizer_specs["params"]["amsgrad"]
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    amsgrad=amsgrad)
            elif optimizer_specs["name"] == "SGD":
                learning_rate = optimizer_specs["params"]["learning_rate"]
                momentum = optimizer_specs["params"]["momentum"]
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=momentum)
            else:
                raise ValueError("Invalid optimizer name")

        # define tb writer
        tb_dir_per_model = os.path.join(
            tb_dir_exp,
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
        model.train()
        for step in tqdm(range(num_train_steps)):
            # load train batch
            try:
                data = next(iterator_train)
            except StopIteration:
                iterator_train = iter(dataloader_train)
                data = next(iterator_train)
            data_device = move_dict_to_device(
                data, device)
            model_input_train = {
                "qpos": data_device[PerMotionDataName.QPOS_GT],
                "qvel": data_device[PerMotionDataName.QVEL_GT]}
            
            # forward pass
            if network != "CRBA":
                optimizer.zero_grad()
            model_output_train = model(model_input_train)

            # compute loss and backprop
            qfrc_net = 0
            for force_name, force_scale in force_specs.items():
                qfrc_net += force_scale * data_device[force_name]
            
            loss_dict = loss_cls.loss(
                model_output_train["inertia"],
                model_input_train["qvel"],
                qfrc_net,
                torch.FloatTensor([dt]).to(device))
            loss = loss_dict["loss"]
            # print(f"inertia[0] has norm: {model_output_train['inertia'][0].norm()}")
            # print(f"impl_pred[0] has norm: {loss_dict['impl_pred'][0].norm()}")
            # print(f"impl_gt[0] has norm: {loss_dict['impl_gt'][0].norm()}")
            # while True:
            #     pass
            if network != "CRBA":
                loss.backward()
                optimizer.step()
            writer.add_scalar("train/loss", loss, step)

            # validate
            # TODO:

        # test
        print(f"Testing {model_name}...")
        # load test batch
        try:
            data = next(iterator_test)
        except StopIteration:
            iterator_test = iter(dataloader_test)
            data = next(iterator_test)
        data_device = move_dict_to_device(
            data, device)
        model_input_test = {
            "qpos": data_device[PerMotionDataName.QPOS_GT],
            "qvel": data_device[PerMotionDataName.QVEL_GT]}
        # forward pass
        model.eval()
        model_output_test = model(model_input_test)
        test_results[model_name] = model_output_test

    # visualize results
    print(f"Visualizing...")
    M_visualizer = InertiaMatrixVisualizer(
        "inertia")
    Ms = np.zeros(
        (len(test_results), len(test_seq_idxs), seq_length, dof, dof))
    times = dt * np.arange(0, seq_length) * 1000
    for i, (model_name, model_output_test) in enumerate(test_results.items()):
        Ms[i] = torch.unsqueeze(
            model_output_test["inertia"], dim=1).cpu().detach().numpy()
    for i, test_seq_idx in enumerate(test_seq_idxs):
        M_visualizer.plot_inertia_as_heatmaps(
            list(test_results.keys()),
            times.astype(np.int32),
            Ms[:, i],
            plot_dir=f"{plot_dir_exp}/seq={test_seq_idx:02d}/",
            plot_first_step_only = True)
        M_visualizer.plot_eigenvals_of_inertia(
            list(test_results.keys()),
            times.astype(np.int32),
            Ms[:, i],
            plot_dir=f"{plot_dir_exp}/seq={test_seq_idx:02d}/",
            plot_first_step_only = True)