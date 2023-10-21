# Author: Guanxiong


import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, normalize
from typing import Dict
from enum import Enum
from Utils.angles import angle_util
from Utils.misc import clean_massMat


class LossName(str, Enum):
    IMPULSE_LOSS = "impulse_loss"
    NORMALIZED_IMPLUSE_LOSS = "normalized_impulse_loss"
    VELOCITY_LOSS = "velocity_loss"
    RIGID_INERTIA_LOSS = "rigid_inertia_loss"
    RIGID_INERTIA_INV_LOSS = "rigid_inertia_inv_loss"
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
    

class RigidInertiaLoss:
    r"""
    Loss that penalizes Euclidean dist between GT rigid inertia and estimated effective inertia.
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
        M_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Parameters:
            M_pred: predicted mass matrix
                (num_sims, dof, dof)
            M_gt: GT rigid mass matrix
                (num_sims, dof, dof)

        Return:
            A dictionary containing:
            - loss: loss averaged across samples
        """
        loss = mse_loss(
            M_pred,
            M_gt,
            reduction=self.reduction)
        return {
            "loss": loss}
    

class RigidInertiaInvLoss:
    r"""
    Loss that penalizes Euclidean dist between GT rigid inertia's inverse and estimated
    effective inertia's inverse.
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
        M_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Parameters:
            M_pred: predicted mass matrix
                (num_sims, dof, dof)
            M_gt: GT rigid mass matrix
                (num_sims, dof, dof)

        Return:
            A dictionary containing:
            - loss: loss averaged across samples
        """
        M_gt_inv = torch.inverse(M_gt)
        M_gt_inv = clean_massMat(M_gt_inv)
        loss = mse_loss(
            M_pred_inv,
            M_gt_inv,
            reduction=self.reduction)
        return {
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
        self.pose_loss_norm = hparams["pose_loss_norm"]
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
        root_pos_diff_norm = torch.linalg.vector_norm(
            root_pos_pred - root_pos_gt, ord=2, dim=-1).view(num_sims, num_steps)
        root_pos_diff_sums = torch.sum(root_pos_diff_norm, dim=-1).view(num_sims,)
        loss_root_pos = torch.mean(root_pos_diff_sums)
        
        # compute loss on root orientation
        # we adapt implementation from compute_ori_loss_quat_quat()
        root_rot_gt = self.angle_util.normalize_vector(
            qpos_gt[:, :, [46, 3, 4, 5]].view(-1, 4)).view(-1, 4)
        root_rot_pred = self.angle_util.normalize_vector(
            qpos_pred[:, :, [46, 3, 4, 5]].view(-1, 4)).view(-1, 4)
        root_rot_gt_mat = self.angle_util.compute_rotation_matrix_from_quaternion(
            root_rot_gt)
        root_rot_gt_mat = self.angle_util.get_44_rotation_matrix_from_33_rotation_matrix(
            root_rot_gt_mat).view(num_sims, num_steps, 4, 4)
        root_rot_pred_mat = self.angle_util.compute_rotation_matrix_from_quaternion(
            root_rot_pred)
        root_rot_pred_mat = self.angle_util.get_44_rotation_matrix_from_33_rotation_matrix(
            root_rot_pred_mat).view(num_sims, num_steps, 4, 4)
        root_rot_diff_norm = torch.linalg.matrix_norm(
            root_rot_pred_mat - root_rot_gt_mat,
            ord="fro", dim=(-2, -1)).view(num_sims, num_steps)
        root_rot_diff_sums = torch.sum(root_rot_diff_norm, dim=-1).view(num_sims,)
        loss_root_rot = torch.mean(root_rot_diff_sums)

        # compute loss on joint angles
        poses_gt = qpos_gt[:, :, 6:-1].view(num_sims, num_steps, 40)
        poses_pred = qpos_pred[:, :, 6:-1].view(num_sims, num_steps, 40)
        poss_diff_norms = torch.linalg.vector_norm(
            poses_pred - poses_gt,
            ord=self.pose_loss_norm, dim=-1).view(num_sims, num_steps)
        poss_diff_sums = torch.sum(poss_diff_norms, dim=-1).view(num_sims,)
        loss_poses = torch.mean(poss_diff_sums)
        
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
        qpos_diff_norm = torch.linalg.vector_norm(
            qpos_pred - qpos_gt,
            ord=2, dim=-1).view(num_sims, num_steps)
        qpos_diff_sums = torch.sum(qpos_diff_norm, dim=-1).view(num_sims,)
        loss = torch.mean(qpos_diff_sums)
        return {
            "loss": loss}