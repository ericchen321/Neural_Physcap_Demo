# Author: Guanxiong


import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
from enum import Enum
import h5py


class PerMotionDataName(str, Enum):
    r"""
    Name of each data in the per-motion dataset.
    """
    QPOS_GT = "qpos_gt"
    QVEL_GT = "qvel_gt"
    BFRC_GR_OPT = "bfrc_gr_opt"
    QFRC_GR_OPT = "qfrc_gr_opt"
    M_RIGID = "M_rigid"
    TAU_OPT = "tau_opt"
    GRAVCOL = "gravcol"
    QPOS_OPT_ITERS = "qpos_opt_iters"
    QVEL_OPT_ITERS = "qvel_opt_iters"
    BFRC_GR_OPT_ITERS = "bfrc_gr_opt_iters"
    QFRC_GR_OPT_ITERS = "qfrc_gr_opt_iters"
    M_RIGID_ITERS = "M_rigid_iters"
    TAU_OPT_ITERS = "tau_opt_iters"
    GRAVCOL_ITERS = "gravcol_iters"


class PerMotionDataset(Dataset):
    r"""
    Consisting of kinematic + dynamic data of sequences from a single motion.
    """

    def __init__(
        self,
        h5_path: str,
        seq_length: int,
        use_per_cycle_data: bool) -> None:
        r"""
        Parameters:
            h5_path: path to the h5 file
            seq_length: length of each sequence
            use_per_cycle_data: whether to use per-dynamic cycle data
        """
        super().__init__()
        data_dict_np = self.read_data_from_h5(h5_path)
        self.seq_length = seq_length
        self.use_per_cycle_data = use_per_cycle_data

        if use_per_cycle_data:
            self.data_names = [
                PerMotionDataName.QPOS_OPT_ITERS,
                PerMotionDataName.QVEL_OPT_ITERS,
                PerMotionDataName.BFRC_GR_OPT_ITERS,
                PerMotionDataName.QFRC_GR_OPT_ITERS,
                PerMotionDataName.M_RIGID_ITERS,
                PerMotionDataName.TAU_OPT_ITERS,
                PerMotionDataName.GRAVCOL_ITERS]
        else:
            self.data_names = [
                PerMotionDataName.QPOS_GT,
                PerMotionDataName.QVEL_GT,
                PerMotionDataName.BFRC_GR_OPT,
                PerMotionDataName.QFRC_GR_OPT,
                PerMotionDataName.M_RIGID,
                PerMotionDataName.TAU_OPT,
                PerMotionDataName.GRAVCOL]
        
        # parse the motion into sequences, each having length `seq_length`.
        # also store the sequences as torch tensors
        self.data = {}

        for data_name in self.data_names:
            data_tch = torch.from_numpy(data_dict_np[data_name])
            self.data[data_name] = torch.stack(
                torch.split(data_tch, seq_length, dim=0)[:-1], dim=0)
        
        print(f"Number of sequences: {self.data[self.data_names[0]].shape[0]}")
        print(f"Sequence length: {self.seq_length}")
        if use_per_cycle_data:
            print(f"Using per-cycle data.")
            print(f"shape of qpos_opt_iters: {self.data[PerMotionDataName.QPOS_OPT_ITERS].shape}")
        else:
            print(f"shape of qpos_gt: {self.data[PerMotionDataName.QPOS_GT].shape}")

    def read_data_from_h5(
        self, h5_path: str) -> Dict[str, np.ndarray]:
        r"""
        Read data from h5 file.

        Parameters:
            h5_path: path to the h5 file

        Return:
            A dictionary containing kinematic + dynamic data of a motion:
            - qpos_gt: ground truth generalized positions (full_seq_length, dof+1)
            - qvel_gt: ground truth generalized velocities (full_seq_length, dof)
            - bfrc_gr_opt: optimized Cartesian GRF/M (full_seq_length, 12)
            - qfrc_gr_opt: optimized generalized GRF/M (full_seq_length, dof)
            - M_rigid: rigid-body mass matrix (full_seq_length dof, dof)
            - tau_opt: optimized generalized joint torques (full_seq_length, dof)
            - gravcol: generalized gravity + Coriolis force (full_seq_length, dof)
            - qpos_opt_iters: optimized generalized positions at each iteration
                (num_dyn_cycles, dof+1)
            - qvel_opt_iters: optimized generalized velocities at each iteration
                (num_dyn_cycles, dof)
            - bfrc_gr_opt_iters: optimized Cartesian GRF/M at each iteration
                (num_dyn_cycles, 12)
            - qfrc_gr_opt_iters: optimized generalized GRF/M at each iteration
                (num_dyn_cycles, dof)
            - M_rigid_iters: rigid-body mass matrix at each iteration
                (num_dyn_cycles, dof, dof)
            - tau_opt_iters: optimized generalized joint torques at each iteration
                (num_dyn_cycles, dof)
            - gravcol_iters: generalized gravity + Coriolis force at each iteration
                (num_dyn_cycles, dof)
        """
        ret_dict = {}
        with h5py.File(h5_path, "r") as data:
            for data_name in PerMotionDataName:
                ret_dict[data_name] = data[data_name][:].astype(np.float32)
        return ret_dict
    
    def __len__(self) -> int:
        return self.data[self.data_names[0]].shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ret_dict = {}
        for data_name in self.data_names:
            ret_dict[data_name] = self.data[data_name][idx]
        return ret_dict