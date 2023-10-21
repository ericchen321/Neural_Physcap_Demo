# Author: Guanxiong
# Train inertia estimator offline, i.e. on a single motion divided into clips,
# while not being a part of the dynamic cycle


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
from inertia_models import define_inertia_estimator
from inertia_trainer import store_and_prune_grad
from inertia_losses import (
    LossName,
    ImpulseLoss,
    VelocityLoss,
    RigidInertiaLoss,
    RigidInertiaInvLoss)
from tqdm import tqdm
from Utils.misc import clean_massMat
from Utils.inertia_utils import move_dict_to_device
from torch.utils.tensorboard import SummaryWriter
from Visualizations.inertia_visualization import InertiaMatrixVisualizer


def parse_force_name(
    force_name_generic: str,
    use_per_cycle_data: bool) -> PerMotionDataName:
    if force_name_generic == "tau":
        if use_per_cycle_data:
            force_name = PerMotionDataName.TAU_OPT_ITERS
        else:
            force_name = PerMotionDataName.TAU_OPT
    elif force_name_generic == "gravcol":
        if use_per_cycle_data:
            force_name = PerMotionDataName.GRAVCOL_ITERS
        else:
            force_name = PerMotionDataName.GRAVCOL
    elif force_name_generic == "grf":
        if use_per_cycle_data:
            force_name = PerMotionDataName.QFRC_GR_OPT_ITERS
        else:
            force_name = PerMotionDataName.QFRC_GR_OPT
    else:
        raise ValueError("Invalid generic force name")
    return force_name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        '--config', type=str, help='path to yaml experiment/data config file', required=True)
    p.add_argument(
        '--device', type=str, help='device for train/val/test, cpu or cuda', required=True)
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
    data_use_opts = config["data_use"]
    use_per_cycle_data = data_use_opts["use_per_cycle_data"]
    force_terms = data_use_opts["force_terms"]
    assert len(force_terms) > 0
    estimation_specs = config["estimation_specs"]
    predict_M_inv = estimation_specs["predict_M_inv"]
    num_train_steps = estimation_specs["num_train_steps"]
    loss_name = estimation_specs["loss"]["name"]
    optimizer_specs = estimation_specs["optimizer"]
    batch_size = estimation_specs["batch_size"]
    steps_til_val = estimation_specs["steps_til_val"]
    visual_specs = config["visual_specs"]
    plot_dir_base = visual_specs["plot_dir"]
    plot_dir_exp = os.path.join(
        plot_dir_base,
        "train_inertia_estimator_offline/",
        f"{config_name}+{motion_name}+t={time_curr}/")
    tb_dir_base = visual_specs["tb_dir"]
    tb_dir_exp = os.path.join(
        tb_dir_base,
        "train_inertia_estimator_offline/",
        f"{config_name}+{motion_name}+t={time_curr}/")
    data_save_specs = config["data_save_specs"]
    data_save_dir_base = data_save_specs["data_save_dir"]
    data_save_dir_exp = os.path.join(
        data_save_dir_base,
        "train_inertia_estimator_offline/",
        f"{config_name}+{motion_name}+t={time_curr}/")
    dof = 46
    n_iters = 6
    dt_dcycle = 0.011
    if use_per_cycle_data:
        dt = dt_dcycle
    else:
        dt = n_iters * dt_dcycle
    device = args.device

    # select names of kinematic/dyn data to use
    if use_per_cycle_data:
        qpos_name = PerMotionDataName.QPOS_OPT_ITERS
        qvel_name = PerMotionDataName.QVEL_OPT_ITERS
        tau_name = PerMotionDataName.TAU_OPT_ITERS
        gravcol_name = PerMotionDataName.GRAVCOL_ITERS
        M_rigid_name = PerMotionDataName.M_RIGID_ITERS
    else:
        qpos_name = PerMotionDataName.QPOS_GT
        qvel_name = PerMotionDataName.QVEL_GT
        tau_name = PerMotionDataName.TAU_OPT
        gravcol_name = PerMotionDataName.GRAVCOL
        M_rigid_name = PerMotionDataName.M_RIGID
    
    # load data
    dataset = PerMotionDataset(h5_path, seq_length, use_per_cycle_data)
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

    # set up validation dataloader
    dataloader_val = DataLoader(
        dataset_val,
        batch_size = len(dataset_val),
        shuffle = True,
        num_workers = 0)
    iterator_val = iter(dataloader_val)
    
    # for each model specified, train + val + test
    test_results = {}
    for model_name, model_specs in estimation_specs["models"].items():
        # get network name
        network = model_specs["network"]

        # define inertia model
        model = define_inertia_estimator(
            model_specs,
            seq_length,
            dof,
            device)
        
        # define the loss class
        if loss_name == LossName.IMPULSE_LOSS:
            assert not predict_M_inv
            loss_cls = ImpulseLoss(dof=dof, reduction="mean")
        elif loss_name == LossName.VELOCITY_LOSS:
            assert predict_M_inv
            loss_cls = VelocityLoss(dof=dof, reduction="mean")
        elif loss_name == LossName.RIGID_INERTIA_LOSS:
            assert not predict_M_inv
            loss_cls = RigidInertiaLoss(
                dof=dof,
                reduction="mean")
        elif loss_name == LossName.RIGID_INERTIA_INV_LOSS:
            assert predict_M_inv
            loss_cls = RigidInertiaInvLoss(
                dof=dof,
                reduction="mean")
        else:
            raise ValueError("Invalid loss name")

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
        for step_idx in tqdm(range(num_train_steps)):
            # load train batch
            try:
                data_train = next(iterator_train)
            except StopIteration:
                iterator_train = iter(dataloader_train)
                data_train = next(iterator_train)
            data_train_device = move_dict_to_device(
                data_train, device)
            model_input_train = {
                "qpos": data_train_device[qpos_name],
                "qvel": data_train_device[qvel_name]}
            
            # forward pass
            if network != "CRBA":
                optimizer.zero_grad()
            model.train()
            model_output_train = model(model_input_train)
            # NOTE: if we're using velocity loss, for CRBA, should take M's inverse
            # for fair comparison
            if network == "CRBA" and predict_M_inv:
                m_inv = torch.inverse(model_output_train["inertia"])
                m_inv = clean_massMat(m_inv)
                model_output_train["inertia"] = m_inv

            # compute loss and backprop
            qfrc_net = 0
            for force_name_generic, force_scale in force_terms.items():
                qfrc_net += force_scale * data_train_device[
                    parse_force_name(force_name_generic, use_per_cycle_data)]
            if loss_name in [LossName.IMPULSE_LOSS, LossName.VELOCITY_LOSS]:
                loss_dict = loss_cls.loss(
                    model_output_train["inertia"],
                    model_input_train["qvel"],
                    qfrc_net,
                    torch.FloatTensor([dt]).to(device))
            elif loss_name in [LossName.RIGID_INERTIA_LOSS, LossName.RIGID_INERTIA_INV_LOSS]:
                loss_dict = loss_cls.loss(
                    model_output_train["inertia"],
                    data_train_device[M_rigid_name][:, 0])
            loss = loss_dict["loss"]
            if network != "CRBA":
                inertia_grads = []
                weights_grads = {}
                inertia_hook = model_output_train["inertia"].register_hook(
                    lambda grad: store_and_prune_grad(
                        grad, inertia_grads, False))
                loss.backward()
                inertia_hook.remove()
                # record grads
                named_params = model.named_parameters()
                for name, params in named_params:
                    assert params.grad is not None
                    if params.requires_grad:
                        weights_grads[name] = params.grad
                optimizer.step()
            
            # log training data to tb
            writer.add_scalar("train_loss", loss, step_idx)
            if network != "CRBA":
                if len(inertia_grads) > 0:
                    if predict_M_inv:
                        scalar_name = "train_grads/M_inv_grads_norm"
                    else:
                        scalar_name = "train_grads/M_grads_norm"
                    writer.add_scalar(
                        scalar_name,
                        torch.linalg.norm(inertia_grads[0]).cpu().detach().numpy(),
                        step_idx)
                for name, param_grads in weights_grads.items():
                    grad_max = torch.max(
                        torch.abs(param_grads)).cpu().detach().numpy()
                    grad_mean = torch.mean(
                        torch.abs(param_grads)).cpu().detach().numpy()
                    writer.add_scalar(
                        f"train_grads/{name}_grads_max",
                        grad_max,
                        step_idx)
                    writer.add_scalar(
                        f"train_grads/{name}_grads_mean",
                        grad_mean,
                        step_idx)

            # validate
            # NOTE: for the time being, use impulse loss as val metric
            if (step_idx+1) % steps_til_val == 0 or step_idx == 0:
                # load val batch
                try:
                    data_val = next(iterator_val)
                except StopIteration:
                    iterator_val = iter(dataloader_val)
                    data_val = next(iterator_val)
                data_val_device = move_dict_to_device(
                    data_val, device)
                model_input_val = {
                    "qpos": data_val_device[qpos_name],
                    "qvel": data_val_device[qvel_name]}
                
                # forward pass
                model.eval()
                with torch.no_grad():
                    model_output_val = model(model_input_val)
                if network == "CRBA" and predict_M_inv:
                    m_inv = torch.inverse(model_output_val["inertia"])
                    m_inv = clean_massMat(m_inv)
                    model_output_val["inertia"] = m_inv
                
                # compute loss as metric
                qfrc_net = 0
                for force_name_generic, force_scale in force_terms.items():
                    qfrc_net += force_scale * data_val_device[
                        parse_force_name(force_name_generic, use_per_cycle_data)]
                if loss_name in [LossName.IMPULSE_LOSS, LossName.VELOCITY_LOSS]:
                    loss_dict = loss_cls.loss(
                        model_output_val["inertia"],
                        model_input_val["qvel"],
                        qfrc_net,
                        torch.FloatTensor([dt]).to(device))
                elif loss_name == LossName.RIGID_INERTIA_LOSS:
                    loss_dict = loss_cls.loss(
                        model_output_val["inertia"],
                        data_val_device[M_rigid_name][:, 0])
                loss_val = loss_dict["loss"]
                writer.add_scalar("val_loss", loss, step_idx)

        # test
        print(f"Testing {model_name}...")
        # load test batch
        try:
            data_test = next(iterator_test)
        except StopIteration:
            iterator_test = iter(dataloader_test)
            data_test = next(iterator_test)
        data_test_device = move_dict_to_device(
            data_test, device)
        model_input_test = {
            "qpos": data_test_device[qpos_name],
            "qvel": data_test_device[qvel_name]}
        # forward pass
        model.eval()
        with torch.no_grad():
            model_output_test = model(model_input_test)
        if network == "CRBA" and predict_M_inv:
            m_inv = torch.inverse(model_output_test["inertia"])
            m_inv = clean_massMat(m_inv)
            model_output_test["inertia"] = m_inv
        test_results[model_name] = model_output_test

        # save data
        data_save_dir_per_model = os.path.join(
            data_save_dir_exp,
            model_name)
        if network != "CRBA":
            os.makedirs(data_save_dir_per_model, exist_ok=True)
            nn_weights_path = f"{data_save_dir_per_model}/{network}.pt"
            torch.save(
                model.state_dict(),
                nn_weights_path)
            print(f"Saved weights of {model_name} to {nn_weights_path}")

    # visualize results
    print(f"Visualizing...")
    if predict_M_inv:
        matrix_name = "inertia_inverse"
    else:
        matrix_name = "inertia"
    M_visualizer = InertiaMatrixVisualizer(
        matrix_name)
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