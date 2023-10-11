# Author: Guanxiong


import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, normalize
from typing import Dict
from enum import Enum


class LossName(str, Enum):
    IMPULSE_LOSS = "impulse_loss"
    NORMALIZED_IMPLUSE_LOSS = "normalized_impulse_loss"
    VELOCITY_LOSS = "velocity_loss"
    SPD_LOSS = "spd_loss"
    MAE_LOSS = "mae_loss"
    MVE_LOSS = "mve_loss"
    MPJPE_LOSS = "mpjpe_loss"


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
        impl_gt = (qfrc_gt[:, 1:] * dt).view(
            num_sims, num_steps-1, self.dof)
        
        # predict qvel_diff
        qvel_diff_pred = torch.bmm(impl_gt, M_pred_inv.transpose(1, 2)).view(
            num_sims, num_steps-1, self.dof)
        # print(f"impl_gt.shape: {impl_gt.shape}")
        # qvel_diff_pred = torch.bmm(
        #     M_pred_inv,
        #     impl_gt.transpose(1,2).view(num_sims, self.dof, num_steps-1))
        
        # compute MSE loss over qvel difference
        # NOTE: we treat data at each step as a sample
        # print(f"norm of qvel_diff_gt, averaged over sim/step: {torch.norm(qvel_diff_gt, dim=(1,2)).mean()}")
        # print(f"norm of qvel_diff_pred, averaged over sim/step: {torch.norm(qvel_diff_pred, dim=(1,2)).mean()}")
        # print(f"norm of M_pred_inv, averaged over sim/step: {torch.norm(M_pred_inv, dim=(1,2)).mean()}")
        # while True:
        #     pass
        loss = mse_loss(
            qvel_diff_pred.reshape(num_sims*(num_steps-1), self.dof),
            qvel_diff_gt.reshape(num_sims*(num_steps-1), self.dof),
            reduction=self.reduction)

        return {
            "qvel_diff_gt": qvel_diff_gt,
            "qvel_diff_pred": qvel_diff_pred,
            "loss": loss}