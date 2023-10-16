# Author: Guanxiong


import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, normalize
from typing import Dict
from enum import Enum
from Utils.angles import angle_util


class LossName(str, Enum):
    IMPULSE_LOSS = "impulse_loss"
    NORMALIZED_IMPLUSE_LOSS = "normalized_impulse_loss"
    VELOCITY_LOSS = "velocity_loss"
    SPD_LOSS = "spd_loss"
    RECON_LOSS_A = "recon_loss_a"
    RECON_LOSS_B = "recon_loss_b"


class ImpulseLoss:
    r"""
    Compute the MSE loss over impulse predictions.
    """
    def __init__(
        self,
        dof = 2,
        reduction: str = 'mean') -> None:
        self.dof = dof
        self.reduction = reduction

    def loss(
        self,
        M_pred: torch.Tensor,
        qvel: torch.Tensor,
        qfrc_gt: torch.Tensor,
        dt: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Parameters:
            M_pred: predicted mass matrix (num_sims, dof, dof)
            qvel: generalized velocities (num_sims, num_steps, dof)
            qfrc_gt: GT forces at each step (num_sims, num_steps, dof)
            dt: timestep size (1,)

        Return:
            A dictionary containing:
            - impl_gt: GT impulse at each step
                (num_sims, num_steps-1, dof)
            - impl_pred: predicted impulse at each step
                (num_sims, num_steps-1, dof)
            - loss: loss across samples
        """
        # compute dq_t - dq_{t-1}
        num_sims, num_steps, _ = qvel.shape
        qvel_diff = (qvel[:, 1:] - qvel[:, :-1]).view(
            num_sims, num_steps-1, self.dof)
        
        # predict impulse
        impl_pred = torch.bmm(qvel_diff, M_pred.transpose(1, 2)).view(
            num_sims, num_steps-1, self.dof)
        
        # compute GT impulse
        impl_gt = (qfrc_gt[:, 1:] * dt).view(
            num_sims, num_steps-1, self.dof)
        
        # compute MSE loss over impulses
        # NOTE: we treat data at each step as a sample
        loss = mse_loss(
            impl_pred.reshape(num_sims*(num_steps-1), self.dof),
            impl_gt.reshape(num_sims*(num_steps-1), self.dof),
            reduction=self.reduction)

        return {
            "impl_gt": impl_gt,
            "impl_pred": impl_pred,
            "loss": loss}
    

class VelocityLoss:
    r"""
    Loss that penalizes the difference between GT and predicted generalized
    velocity difference.
    """
    def __init__(
        self,
        dof = 2,
        reduction: str = 'mean') -> None:
        self.dof = dof
        self.reduction = reduction

    def loss(
        self,
        M_pred_inv: torch.Tensor,
        qvel: torch.Tensor,
        qfrc_gt: torch.Tensor,
        dt: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Parameters:
            M_pred_inv: predicted mass matrix inverse (num_sims, dof, dof)
            qvel: generalized velocities (num_sims, num_steps, dof)
            qfrc_gt: GT forces at each step (num_sims, num_steps, dof)
            dt: timestep size (1,)

        Return:
            A dictionary containing:
            - qvel_diff_gt: GT qvel diff at each step
                (num_sims, num_steps-1, dof)
            - qvel_diff_pred: predicted qvel diff at each step
                (num_sims, num_steps-1, dof)
            - loss: loss across samples
        """
        # compute GT qvel_diff
        num_sims, num_steps, _ = qvel.shape
        qvel_diff_gt = (qvel[:, 1:] - qvel[:, :-1]).view(
            num_sims, num_steps-1, self.dof)
        
        # compute GT impulse
        impl_gt = (qfrc_gt[:, :-1] * dt).view(
            num_sims, num_steps-1, self.dof)
        
        # predict qvel_diff
        qvel_diff_pred = torch.bmm(impl_gt, M_pred_inv.transpose(1, 2)).view(
            num_sims, num_steps-1, self.dof)
        
        # compute MSE loss over qvel difference
        # NOTE: we treat data at each step as a sample
        loss = mse_loss(
            qvel_diff_pred.reshape(num_sims*(num_steps-1), self.dof),
            qvel_diff_gt.reshape(num_sims*(num_steps-1), self.dof),
            reduction=self.reduction)

        return {
            "qvel_diff_gt": qvel_diff_gt,
            "qvel_diff_pred": qvel_diff_pred,
            "loss": loss}
    

class ReconLossA:
    r"""
    Loss that penalizes 1) Euclidean distance between GT and predicted joint
    poses across samples and steps; 2) Euclidean distance between GT and predicted
    root positions; 3) Angular distance between GT and predicted root orientations.
    """

    def __init__(
        self,
        hparams: Dict,
        dof: int = 47) -> None:
        self.dof = dof
        assert self.dof == 47, "dof must be 47"
        self.weight_root_pos = hparams["weight_root_pos"]
        self.weight_root_rot = hparams["weight_root_rot"]
        self.weight_poses = hparams["weight_poses"]
        self.angle_util = angle_util()

    def loss(
        self,
        qpos_pred: torch.Tensor,
        qpos_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Paremters:
            qpos_pred: predicted joint pose (num_sims, num_steps, dof)
            qpos_gt: GT joint pose (num_sims, num_steps, dof)

        Return:
            A dictionary containing:
            - loss: MSE loss across samples and steps
        """
        # precondition checks
        assert qpos_pred.shape == qpos_gt.shape
        num_sims, num_steps, dof = qpos_pred.shape
        assert dof == self.dof

        # compute loss on root pos
        root_pos_gt = qpos_gt[:, :, :3]
        root_pos_pred = qpos_pred[:, :, :3]
        root_pos_diff_norm = torch.norm(
            root_pos_pred - root_pos_gt, p="fro", dim=-1).view(num_sims, num_steps)
        root_pos_diff_sums = torch.sum(root_pos_diff_norm, dim=-1).view(num_sims,)
        loss_root_pos = torch.mean(root_pos_diff_sums)
        
        # compute cosine distance on root orientation
        # NOTE: We use Eq. 7 from Eric Gartner's 2022 paper (except that
        # here we apply it only to root rot)
        eps = 1e-9
        root_rot_gt = self.angle_util.normalize_vector(
            qpos_gt[:, :, [46, 3, 4, 5]].view(-1, 4)).view(-1, 4)
        root_rot_pred = self.angle_util.normalize_vector(
            qpos_pred[:, :, [46, 3, 4, 5]].view(-1, 4)).view(-1, 4)
        root_rot_diff = torch.acos(
            torch.abs(
                torch.sum(root_rot_gt * root_rot_pred, dim=-1)).clamp(
                    -1.0+eps, 1.0-eps)).view(num_sims, num_steps)
        root_rot_diff_sums = torch.sum(root_rot_diff, dim=-1).view(num_sims,)
        loss_root_rot = torch.mean(root_rot_diff_sums)

        # compute L1 loss on angles
        poses_gt = qpos_gt[:, :, 6:-1].view(num_sims, num_steps, 40)
        poses_pred = qpos_pred[:, :, 6:-1].view(num_sims, num_steps, 40)
        poss_diff_norms = torch.norm(
            poses_pred - poses_gt, p=1, dim=-1).view(num_sims, num_steps)
        poss_diff_sums = torch.sum(poss_diff_norms, dim=-1).view(num_sims,)
        loss_poses = torch.mean(poss_diff_sums)

        # print(f"loss_root_pos: {loss_root_pos}")
        # print(f"loss_root_rot: {loss_root_rot}")
        # print(f"loss_poses: {loss_poses}")
        # while True:
        #     pass

        # if torch.any(torch.isnan(loss_root_pos)):
        #     print("loss_root_pos is nan")
        #     while True:
        #         pass
        # if torch.any(torch.isnan(loss_root_rot)):
        #     print("loss_root_rot is nan")
        #     print(torch.abs(torch.sum(
        #         root_rot_gt * root_rot_pred, dim=-1)))
        #     while True:
        #         pass
        # if torch.any(torch.isnan(loss_poses)):
        #     print("loss_poses is nan")
        #     while True:
        #         pass
        
        loss = self.weight_root_pos * loss_root_pos + \
            self.weight_root_rot * loss_root_rot + \
            self.weight_poses * loss_poses

        return {
            "loss_root_pos": loss_root_pos,
            "loss_root_rot": loss_root_rot,
            "loss_poses": loss_poses,
            "loss": loss}
    
class ReconLossB:
    r"""
    Loss that penalizes Euclidean distance between GT and predicted qpos.
    """

    def __init__(
        self,
        dof: int = 47,
        reduction: str = 'mean') -> None:
        self.dof = dof
        assert self.dof == 47, "dof must be 47"
        self.reduction = reduction

    def loss(
        self,
        qpos_pred: torch.Tensor,
        qpos_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Paremters:
            qpos_pred: predicted joint pose (num_sims, num_steps, dof)
            qpos_gt: GT joint pose (num_sims, num_steps, dof)

        Return:
            A dictionary containing:
            - loss: MSE loss across samples and steps
        """
        # precondition checks
        assert qpos_pred.shape == qpos_gt.shape
        num_sims, num_steps, dof = qpos_pred.shape
        assert dof == self.dof

        # sum up error norms across steps, similar to Eq. 11 from
        # "Physically Plausible Reconstruction from Monocular Videos"
        # then average across sequences
        qpos_diff_norm = torch.norm(
            qpos_pred - qpos_gt, p="fro", dim=-1).view(num_sims, num_steps)
        qpos_diff_sums = torch.sum(qpos_diff_norm, dim=-1).view(num_sims,)
        loss = torch.mean(qpos_diff_sums)
        return {
            "loss": loss}