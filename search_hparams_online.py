# Author: Guanxiong


import argparse
import glob
import os
import yaml
from typing import Dict, List
from train_inertia_estimator_online import train_inertia_estimator_online
from Visualizations.inertia_visualization import write_config_to_file
import traceback


def generate_configs(
    config_base: Dict,
    config_suffix: str,
    write_flag: bool = True) -> List:
    # define hparams to change
    learning_rates = [5.0e-4, 1.0e-4, 5.0e-5, 1.0e-5, 5.0e-6]
    seq_lengths = [10, 20]
    weight_combinations = [
        {
            "weight_root_pos": 1.0,
            "weight_root_rot": 1.0,
            "weight_poses": 1.0},
        {
            "weight_root_pos": 10.0,
            "weight_root_rot": 5.0,
            "weight_poses": 1.0},
        {
            "weight_root_pos": 1.0,
            "weight_root_rot": 0.5,
            "weight_poses": 0.1},
        {
            "weight_root_pos": 10.0,
            "weight_root_rot": 10.0,
            "weight_poses": 1.0},
        {
            "weight_root_pos": 1.0,
            "weight_root_rot": 1.0,
            "weight_poses": 0.1}]
    for wc in weight_combinations:
        wc["pose_loss_norm"] = 2
    mlp_widths = [
        [8192, 4096, 2048],
        [2048, 2048, 2048],
        [4096, 4096, 2048, 2048]]
    
    # define paths to pretrained inertia estimators
    # NOTE: one-to-one correspondence with mlp_widths
    pretrained_weights_paths = [
        "config_g02+sample_dance+t=2023-10-19-17-29-01/",
        "config_g06+sample_dance+t=2023-10-23-18-00-38/",
        "config_g04+sample_dance+t=2023-10-19-17-44-24/"]
    
    # define dir to save configs
    config_save_path = os.path.dirname(args.base_config)
    
    # generate configs
    config_idx = 0
    configs = []
    for lr_idx, lr in enumerate(learning_rates):

        for seq_length_idx, seq_length in enumerate(seq_lengths):

            for wc_idx, wc in enumerate(weight_combinations):

                for mlp_width_idx, mlp_width in enumerate(mlp_widths):

                    # define config name
                    config_name = "".join([
                        f"config_{config_suffix}{config_idx:03d}",
                        f"+lr_idx={lr_idx:02d}",
                        f"+seq_length_idx={seq_length_idx:02d}",
                        f"+wc_idx={wc_idx:02d}",
                        f"+mlp_width_idx={mlp_width_idx:02d}"])
                    
                    # derive config from base config
                    config = config_base.copy()
                    config["config_name"] = config_name
                    config["seq_length"] = seq_length
                    config[
                        "inertia_estimate"][
                        "pretrained_weights"][
                        "experiment_name"] = pretrained_weights_paths[mlp_width_idx]
                    config[
                        "inertia_estimate"][
                        "loss"][
                        "hparams"] = wc
                    config[
                        "inertia_estimate"][
                        "optimizer"][
                        "params"][
                        "learning_rate"] = lr
                    for _, model_specs in config["inertia_estimate"]["models"].items():
                        if model_specs["network"] != "CRBA":
                            model_specs["mlp"]["widths"] = []
                            for per_layer_width in mlp_width:
                                model_specs["mlp"]["widths"].append(per_layer_width)

                    # save config
                    if write_flag:
                        write_config_to_file(config, config_save_path, f"{config_name}.yaml")
                    configs.append({
                        "config_idx": config_idx,
                        "config_name": config_name,
                        "config_path": f"{config_save_path}/{config_name}.yaml",
                        "config": config})
                    
                    config_idx += 1
    if write_flag:
        print(f"Defined and saved {config_idx} configs")
    else:
        print(f"Defined {config_idx} configs")
    return configs


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_gen_configs = subparsers.add_parser("generate_configs")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument(
        '--start_idx', type=int, required=True)
    parser_train.add_argument(
        '--end_idx', type=int, required=True)
    parser_train.add_argument(
        '--device', type=str, help='device for train/val/test, cpu or cuda', required=True)
    
    parser.add_argument(
        '--base_config', type=str, help='path to base yaml config file', required=True)
    parser.add_argument(
        '--config_suffix', type=str, help='can be a letter or sth', required=True)

    args = parser.parse_args()

    # read base config
    with open(args.base_config, "r") as stream:
        try:
            config_base = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    # generate configs
    if args.command == "generate_configs":
        generate_configs(
            config_base,
            config_suffix=args.config_suffix,
            write_flag = True)

    # train
    elif args.command == "train":
        config_meta_dicts = generate_configs(
            config_base,
            config_suffix=args.config_suffix,
            write_flag = False)
        num_configs = len(config_meta_dicts)
        
        # check indices
        if args.start_idx < 0 or args.start_idx >= num_configs:
            raise ValueError("start_idx out of range")
        if args.end_idx < 0 or args.end_idx >= num_configs:
            raise ValueError("end_idx out of range")
        
        # define search name
        search_name = os.path.basename(
            os.path.normpath(os.path.dirname(args.base_config)))
        
        # train, validate
        for config_meta_dict in config_meta_dicts:
            # extract 3-dig config index
            config_idx = config_meta_dict["config_idx"]

            # check if in range
            if config_idx >= args.start_idx and config_idx <= args.end_idx:
                config_path = config_meta_dict["config_path"]
                print(f"Training config {config_path}...")
                with open(config_path, "r") as stream:
                    try:
                        config = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
                        config = None
                try:
                    train_inertia_estimator_online(
                        config, device=args.device,
                        nn_name_suffix=f"config_{args.config_suffix}{config_idx:03d}",
                        exp_group="search_hparams_online",
                        exp_subgroup=search_name)
                except RuntimeError:
                    print(f"Runtime error in training config {config_path}")
                    traceback.print_exc()
                    continue