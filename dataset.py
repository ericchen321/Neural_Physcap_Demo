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


class PerMotionDataset(Dataset):
    r"""
    Consisting of kinematic + dynamic data of sequences from a single motion.
    """

    def __init__(
        self,
        h5_path: str,
        seq_length: int) -> None:
        r"""
        Parameters:
            h5_path: path to the h5 file
            seq_length: length of each sequence
        """
        super().__init__()
        data_dict_np = self.read_data_from_h5(h5_path)
        self.seq_length = seq_length
        
        # parse the motion into sequences, each having length `seq_length`.
        # also store the sequences as torch tensors
        self.full_seq_length = data_dict_np[PerMotionDataName.QPOS_GT].shape[0]
        for data_name, data_np in data_dict_np.items():
            data_tch = torch.from_numpy(data_np)
            if data_name == PerMotionDataName.QPOS_GT:
                self.qpos_gt = torch.stack(
                    torch.split(data_tch, seq_length, dim=0)[:-1], dim=0)
            elif data_name == PerMotionDataName.QVEL_GT:
                self.qvel_gt = torch.stack(
                    torch.split(data_tch, seq_length, dim=0)[:-1], dim=0)
            elif data_name == PerMotionDataName.BFRC_GR_OPT:
                self.bfrc_gr_opt = torch.stack(
                    torch.split(data_tch, seq_length, dim=0)[:-1], dim=0)
            elif data_name == PerMotionDataName.QFRC_GR_OPT:
                self.qfrc_gr_opt = torch.stack(
                    torch.split(data_tch, seq_length, dim=0)[:-1], dim=0)
            elif data_name == PerMotionDataName.M_RIGID:
                self.M_rigid = torch.stack(
                    torch.split(data_tch, seq_length, dim=0)[:-1], dim=0)
            elif data_name == PerMotionDataName.TAU_OPT:
                self.tau_opt = torch.stack(
                    torch.split(data_tch, seq_length, dim=0)[:-1], dim=0)
            elif data_name == PerMotionDataName.GRAVCOL:
                self.gravcol = torch.stack(
                    torch.split(data_tch, seq_length, dim=0)[:-1], dim=0)
        self.num_seqs = self.qpos_gt.shape[0]
        print(f"Number of sequences: {self.num_seqs}")
        print(f"Full sequence length: {self.full_seq_length}")
        print(f"Sequence length: {self.seq_length}")
        print(f"shape of qpos_gt: {self.qpos_gt.shape}")
        print(f"shape of M_rigid: {self.M_rigid.shape}")

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
        """
        ret_dict = {}
        with h5py.File(h5_path, "r") as data:
            ret_dict[PerMotionDataName.QPOS_GT] = data[
                PerMotionDataName.QPOS_GT][:].astype(np.float32)
            ret_dict[PerMotionDataName.QVEL_GT] = data[
                PerMotionDataName.QVEL_GT][:].astype(np.float32)
            ret_dict[PerMotionDataName.BFRC_GR_OPT] = data[
                PerMotionDataName.BFRC_GR_OPT][:].astype(np.float32)
            ret_dict[PerMotionDataName.QFRC_GR_OPT] = data[
                PerMotionDataName.QFRC_GR_OPT][:].astype(np.float32)
            ret_dict[PerMotionDataName.M_RIGID] = data[
                PerMotionDataName.M_RIGID][:].astype(np.float32)
            ret_dict[PerMotionDataName.TAU_OPT] = data[
                PerMotionDataName.TAU_OPT][:].astype(np.float32)
            ret_dict[PerMotionDataName.GRAVCOL] = data[
                PerMotionDataName.GRAVCOL][:].astype(np.float32)
        return ret_dict
    
    def __len__(self) -> int:
        return self.qpos_gt.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            PerMotionDataName.QPOS_GT: self.qpos_gt[idx],
            PerMotionDataName.QVEL_GT: self.qvel_gt[idx],
            PerMotionDataName.BFRC_GR_OPT: self.bfrc_gr_opt[idx],
            PerMotionDataName.QFRC_GR_OPT: self.qfrc_gr_opt[idx],
            PerMotionDataName.M_RIGID: self.M_rigid[idx],
            PerMotionDataName.TAU_OPT: self.tau_opt[idx],
            PerMotionDataName.GRAVCOL: self.gravcol[idx]
        }