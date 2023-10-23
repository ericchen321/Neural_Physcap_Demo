# Author: Guanxiong and the original authors
# Train inertia estimator online, i.e. on a single motion undivided,
# while being a part of the dynamic cycle


import os 
dir_path = os.path.dirname(os.path.realpath(__file__)) 
import numpy as np
from dataset import PerMotionExtendedDataset
from inertia_trainer import OnlineTrainer
from torch.utils.data import random_split
import argparse  
from datetime import datetime
import yaml
        
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='arguments for predictions')
    parser.add_argument('--img_width', type=float, default=1280)
    parser.add_argument('--img_height', type=float, default=720)
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--device', type=str, help='device for train/val/test, cpu or cuda', required=True)
    args = parser.parse_args()

    # extract configs
    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    delta_t = config["dynamics"]["delta_t"]
    urdf_path = config["dynamics"]["humanoid_urdf_path"]
    net_path = config["pretrained_nw_path"]
    img_width = config["kinematics"]["img_width"]
    img_height = config["kinematics"]["img_height"]
    if config["kinematics"]["floor_known"]:
        RT = np.load(config["kinematics"]["floor_position_path"]) 
    else:
        RT = None
    if config["kinematics"]["cam_params_known"]:  
        K = np.load(config["kinematics"]["cam_params_path"]) 
    else: 
        K = np.array(
            [1000, 0, img_width/2, 0, 0, 1000, img_height/2, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4)
    config_name = config["config_name"]
    h5_path = config["h5_path"]
    motion_name = os.path.splitext(
        os.path.basename(h5_path))[0].split("seq_name=")[1]
    time_curr = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    seq_length = config["seq_length"]
    M_est_specs = config["inertia_estimate"]
    predict_M_inv = M_est_specs["predict_M_inv"]
    if "pretrained_weights" in M_est_specs:
        pretrained_weights_specs = M_est_specs["pretrained_weights"]
    else:
        pretrained_weights_specs = None
    num_train_steps = M_est_specs["num_train_steps"]
    loss_specs = M_est_specs["loss"]
    optimizer_specs = M_est_specs["optimizer"]
    batch_size = M_est_specs["batch_size"]
    steps_til_val = M_est_specs["steps_til_val"]
    visual_specs = config["visual_specs"]
    plot_dir_base = visual_specs["plot_dir"]
    plot_dir_exp = os.path.join(
        plot_dir_base,
        "train_inertia_estimator_online/",
        f"{config_name}+{motion_name}+t={time_curr}/")
    tb_dir_base = visual_specs["tb_dir"]
    tb_dir_exp = os.path.join(
        tb_dir_base,
        "train_inertia_estimator_online/",
        f"{config_name}+{motion_name}+t={time_curr}/")
    data_save_specs = config["data_save_specs"]
    data_save_dir_base = data_save_specs["data_save_dir"]
    data_save_dir_exp = os.path.join(
        data_save_dir_base,
        "train_inertia_estimator_online/",
        f"{config_name}+{motion_name}+t={time_curr}/")
    dof = 46
    temporal_window = config["temporal_window"]
    p_2ds_path = config["kinematics"]["2d_joint_pos_path"]
    num_dyn_cycles = config["dynamics"]["num_dyn_cycles"]
    con_thresh = config["dynamics"]["con_thresh"]
    tau_limit = config["dynamics"]["tau_limit"]
    speed_limit = config["kinematics"]["speed_limit"]
    device = args.device

    # load data
    # NOTE: temporal_window_ori is the original temporal window at the time
    # when the dataset was sampled
    temporal_window_ori = 10
    dataset = PerMotionExtendedDataset(
        h5_path,
        p_2ds_path,
        temporal_window_ori,
        seq_length)
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

    for model_name, model_specs in M_est_specs["models"].items():
        # define save paths per model
        tb_dir_per_model = os.path.join(
            tb_dir_exp,
            model_name)
        plot_dir_per_model = os.path.join(
            plot_dir_exp,
            model_name)
        data_save_dir_per_model = os.path.join(
            data_save_dir_exp,
            model_name)

        # train in the dynamic cycle
        print(f"Training {model_name}...")
        trainer = OnlineTrainer(
            urdf_path,
            net_path,
            seq_length,
            dataset_train,
            dataset_val,
            data_save_dir_per_model,
            tb_dir_per_model,
            model_name,
            model_specs,
            predict_M_inv,
            pretrained_weights_specs,
            loss_specs,
            optimizer_specs,
            img_width, img_height,
            K, RT,
            neural_PD = True,
            num_dyn_cycles = num_dyn_cycles,
            delta_t = delta_t,
            temporal_window = temporal_window,
            con_thresh = con_thresh,
            limit = tau_limit,
            speed_limit = speed_limit,
            motion_name = motion_name,
            batch_size = batch_size,
            num_train_steps = num_train_steps,
            steps_til_val = steps_til_val,
            device = device)
        trainer.train_and_validate()
        trainer.save_model()