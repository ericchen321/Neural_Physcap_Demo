# Author: Guanxiong and the original authors


import torch
import os 
import rbdl 
import tqdm
import os 
dir_path = os.path.dirname(os.path.realpath(__file__)) 
import numpy as np
from networks import (
    ContactEstimationNetwork,
    DynamicNetwork,
    GRFNet)
from dataset import (
    PerMotionDataName,
    PerMotionExtendedDataName)
from torch.utils.data import DataLoader
from inertia_models import define_inertia_estimator
from inertia_losses import (
    LossName,
    ReconLossA,
    ReconLossB)
from Utils.angles import angle_util 
import Utils.misc as ut
import Utils.phys as ppf
from Utils.core_utils import CoreUtils
from Utils.initializer import InitializerConsistentHumanoid2
from Utils.inertia_utils import move_dict_to_device
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List


def store_and_clip_grad(
    grad,
    storing_list,
    clip_grad: bool = False,
    grad_thresh: float = 4.0):
    if clip_grad:
        grad_new = torch.clamp(grad, -grad_thresh, grad_thresh)
        storing_list.append(grad_new.clone())
        return grad_new
    else:
        storing_list.append(grad.clone())


def store_and_prune_grad(
    grad,
    storing_list,
    prune_grad: bool = False,
    grad_thresh: float = 4.0):
    if prune_grad:
        if torch.isnan(grad).any():
            print("nan in grad")
        grad_new = torch.nan_to_num(
            grad, nan=0.0, posinf=grad_thresh, neginf=-grad_thresh)
        storing_list.append(grad_new.clone())
        return grad_new
    else:
        storing_list.append(grad.clone())


class Trainer():
    r"""
    A trainer for an inertia estimator.
    """
    
    def __init__(
        self,
        urdf_path,
        net_path,
        seq_length,
        dataset_train,
        save_base_path,
        tb_dir_path,
        inertia_model_name,
        inertia_model_specs,
        predict_M_inv,
        pretrained_weights_specs,
        loss_specs,
        optimizer_specs,
        w, h, K, RT,
        neural_PD = True,
        num_dyn_cycles = 6,
        delta_t = 0.011,
        temporal_window = 10,
        con_thresh = 0.01,
        limit = 50,
        speed_limit = 35.0,
        motion_name = "",
        batch_size = 1,
        num_train_steps = 1000,
        device = "cuda"):

        # save basic configs
        self.seq_length = seq_length
        self.w = w
        self.h = h
        self.neural_PD = neural_PD
        self.num_dyn_cycles = num_dyn_cycles
        self.delta_t = delta_t
        self.temporal_window = temporal_window
        self.con_thresh = con_thresh
        self.limit = limit
        self.speed_limit = speed_limit
        self.save_base_path = save_base_path 
        self.motion_name = motion_name
        self.batch_size = batch_size
        self.num_train_steps = num_train_steps
        self.grad_thresh = 4.0
        self.device = device
        self.num_joints = 24

        # initialize utilities
        self.cu = CoreUtils(45, delta_t)
        self.au = angle_util()
        
        # save joint mapping configs
        self.openpose_dic2 = {
            "base": 7,
            "left_hip": 11,
            "left_knee": 12,
            "left_ankle": 13,
            "left_toe": 19,
            "right_hip": 8,
            "right_knee": 9,
            "right_ankle": 10,
            "right_toe": 22,
            "neck": 0,
            "head": 14,
            "left_shoulder": 4,
            "left_elbow": 5,
            "left_wrist": 6,
            "right_shoulder": 1,
            "right_elbow": 2,
            "right_wrist": 3 }
        self.target_joints = [
            "head",
            "neck",
            "left_hip",
            "left_knee",
            "left_ankle",
            "left_toe",
            "right_hip",
            "right_knee",
            "right_ankle",
            "right_toe",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "right_shoulder",
            "right_elbow",
            "right_wrist"]
        self.target_ids = [self.openpose_dic2[key] for key in self.target_joints]

        # load humanoid model and initialize stuff
        self.rbdl_model = rbdl.loadModel(urdf_path.encode(), floating_base=True)
        humanoid_info = InitializerConsistentHumanoid2(self.batch_size, self.target_joints)
        self.rbdl_dic = humanoid_info.get_rbdl_dic()
        self.target_joint_ids = humanoid_info.get_target_joint_ids()
        self.model_addresses = {"0": self.rbdl_model, "1": self.rbdl_model}
        self.dof = self.rbdl_model.qdot_size

        # build and load pretrained models
        self.ConNet = ContactEstimationNetwork(
            in_channels = 2*len(self.target_ids),
            num_features = 1024,
            out_channels = 4,
            num_blocks = 4).to(self.device)
        self.GRFNet = GRFNet(
            input_dim = 577,
            output_dim = self.dof + self.dof+3*4).to(self.device)
        self.DyNet = DynamicNetwork(
            input_dim = 2302,
            output_dim = self.dof,
            offset_coef = 10).to(self.device)
        if os.path.exists(net_path + "ConNet.pkl"): 
            self.ConNet.load_state_dict(
                torch.load(net_path + "ConNet.pkl", map_location=torch.device(self.device)))
        else:
            raise FileNotFoundError('pretrained ConNet not found')
        if os.path.exists(net_path + "GRFNet.pkl"):
            self.GRFNet.load_state_dict(
                torch.load(net_path + "GRFNet.pkl" ,map_location=torch.device(self.device)))
        else:
            raise FileNotFoundError('pretrained GRFNet not found')
        if os.path.exists(net_path + "DyNet.pkl"):
            self.DyNet.load_state_dict(
                torch.load(net_path+ "DyNet.pkl", map_location=torch.device(self.device)))
        else:
            raise FileNotFoundError('pretrained DyNet not found')
        self.ConNet.eval()
        self.DyNet.eval()
        self.GRFNet.eval()

        # define inertia estimator
        self.inertia_model_name = inertia_model_name
        self.inertia_estimator_specs = inertia_model_specs
        self.predict_M_inv = predict_M_inv
        self.inertia_estimator = define_inertia_estimator(
            self.inertia_estimator_specs,
            1,
            self.dof,
            device=self.device)
        
        # load pretrained inertia estimator
        if self.inertia_estimator_specs['network'] != "CRBA" and \
            pretrained_weights_specs is not None:
            model_weights_path = os.path.join(
                "data_logging/",
                "train_inertia_estimator_offline/",
                pretrained_weights_specs["experiment_name"],
                f"{self.inertia_model_name}/",
                f"{self.inertia_estimator_specs['network']}.pt")
            try:
                self.inertia_estimator.load_state_dict(
                    torch.load(
                        model_weights_path, map_location=torch.device(self.device)),
                    strict = True)
                print(f"Loaded {self.inertia_model_name} model for inertia estimation")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find {self.inertia_model_name} model for inertia estimation")

        # setup custom pytorch functions including the Physics model 
        self.PyFK = ppf.PyForwardKinematicsQuaternion().apply 
        self.PyFD = ppf.PyForwardDynamics.apply

        # set up dataloader
        self.dataloader_train = DataLoader(
            dataset_train,
            batch_size = self.batch_size,
            shuffle = True,
            drop_last=True,
            num_workers = 0)
        
        # define the loss class
        self.loss_name = loss_specs["name"]
        if self.loss_name == LossName.RECON_LOSS_A:
            hparams = loss_specs["hparams"]
            self.loss_cls = ReconLossA(
                hparams,
                dof = self.dof+1)
        elif self.loss_name == LossName.RECON_LOSS_B:
            self.loss_cls = ReconLossB(
                dof = self.dof+1,
                reduction = "mean")
        else:
            raise ValueError("Invalid loss name")
        
        # define optimizer
        if self.inertia_estimator_specs["network"] != "CRBA":
            if optimizer_specs["name"] == "Adam":
                learning_rate = optimizer_specs["params"]["learning_rate"]
                amsgrad = optimizer_specs["params"]["amsgrad"]
                self.optimizer = torch.optim.Adam(
                    self.inertia_estimator.parameters(),
                    lr=learning_rate,
                    amsgrad=amsgrad)
            elif optimizer_specs["name"] == "SGD":
                learning_rate = optimizer_specs["params"]["learning_rate"]
                momentum = optimizer_specs["params"]["momentum"]
                self.optimizer = torch.optim.SGD(
                    self.inertia_estimator.parameters(),
                    lr=learning_rate,
                    momentum=momentum)
            else:
                raise ValueError("Invalid optimizer name")

        # pre-compute some kinematic data
        self.RT = RT
        self.Rs = torch.FloatTensor(
            self.RT[:3, :3]).to(self.device).expand(self.batch_size, -1, -1).view(
                self.batch_size, 3, 3)

        # set up TB writer
        self.writer = SummaryWriter(log_dir = tb_dir_path)

    def get_grav_corioli(
        self,
        sub_ids: List[int],
        floor_normals_in: torch.Tensor,
        q_in: torch.Tensor,
        qdot_in: torch.Tensor,
        device: str = "cuda") -> torch.Tensor:
        batch_size, _ = q_in.shape
        q = q_in.cpu().detach().numpy().astype(float)
        qdot = qdot_in.cpu().detach().numpy().astype(float)
        gcc = np.zeros(
            (batch_size, self.rbdl_model.qdot_size))
        floor_normals = floor_normals_in.cpu().detach().numpy()
        for batch_id in range(batch_size):
            sid = sub_ids[batch_id]
            model_address = self.model_addresses[str(int(sid))]
            model_address.gravity = -9.8 * floor_normals[batch_id]
            rbdl.InverseDynamics(
                model_address,
                q[batch_id],
                qdot[batch_id],
                np.zeros(self.rbdl_model.qdot_size).astype(float),
                gcc[batch_id])
        return torch.FloatTensor(gcc).to(device)

    def contact_label_estimation(
        self, input_rr):
        pred_labels = self.ConNet(input_rr)
        pred_labels = pred_labels.to(input_rr.device)
        pred_labels_prob = pred_labels.clone()
        pred_labels[pred_labels < self.con_thresh] = 0
        pred_labels[pred_labels >= self.con_thresh] = 1
        return pred_labels, pred_labels_prob

    def train_and_validate(
        self):

        losses = []
        with torch.autograd.set_detect_anomaly(True):
            iterator_train = iter(self.dataloader_train)
            for train_step_idx in tqdm.tqdm(list(range(self.num_train_steps))):
                # load train batch
                try:
                    data_train = next(iterator_train)
                except StopIteration:
                    iterator_train = iter(self.dataloader_train)
                    data_train = next(iterator_train)
                data_train_per_batch = move_dict_to_device(
                    data_train, self.device)
                
                # compute normalized target 2D joint positions
                p_2ds = data_train_per_batch[
                    PerMotionExtendedDataName.P_2DS_GT].view(
                    self.batch_size, self.seq_length, self.num_joints, 2)
                p_2d_base = p_2ds[:, :, self.openpose_dic2["base"]].view(
                    self.batch_size, self.seq_length, 2)
                p_2ds = p_2ds[:, :, self.target_ids].view(
                    self.batch_size, self.seq_length, len(self.target_ids), 2)
                p_2ds[:, :, :, 0] /= self.w
                p_2ds[:, :, :, 1] /= self.h
                p_2d_base[:, :, 0] /= self.w
                p_2d_base[:, :, 1] /= self.h
                p_2ds_rr = (p_2ds - p_2d_base.unsqueeze(dim=2)).view(
                    self.batch_size, self.seq_length, len(self.target_ids), 2)
                
                # simulate sequences in a batch altogether
                loss_per_step = []
                qpos_pred = torch.zeros(
                    self.batch_size, self.seq_length-self.temporal_window, self.dof+1,
                    dtype=torch.float32,
                    device=self.device)
                qvel_pred = torch.zeros(
                    self.batch_size, self.seq_length-self.temporal_window, self.dof,
                    dtype=torch.float32,
                    device=self.device)
                if self.inertia_estimator_specs["network"] != "CRBA":
                    inertia_grads = []
                    weights_grads = {}
                for sim_step_idx in range(self.temporal_window, self.seq_length):
                    # get data per step
                    data_train_per_step: Dict[str, torch.Tensor] = {}
                    for data_name, data in data_train_per_batch.items():
                        data_train_per_step[data_name] = data[:, [sim_step_idx]]

                    # set axis vectors
                    # basis_vec_w = torch.FloatTensor(
                    #     np.array(
                    #         [[1, 0, 0, ], [0, 1, 0, ], [0, 0, 1, ]])).to(self.device).view(1, 3, 3)
                    # basis_vec_w = basis_vec_w.expand(self.batch_size, -1, -1)
                    
                    # setup input_rr (for contact label estimation) and floor normals
                    # for computing forces
                    # frame_rr_2Ds = p_2ds_rr.clone()[
                    #     :,
                    #     sim_step_idx-self.temporal_window:sim_step_idx].view(
                    #     self.batch_size, self.temporal_window, len(self.target_ids)*2)
                    # floor_normals = torch.transpose(
                    #     torch.bmm(
                    #         self.Rs,
                    #         torch.transpose(
                    #             basis_vec_w, 1, 2)), 1, 2)[:, 1].view(self.batch_size, 3)
                    # input_rr = frame_rr_2Ds.view(
                    #     self.batch_size, self.temporal_window, -1)

                    # extract target pose
                    q_tar = data_train_per_step[
                        PerMotionDataName.QPOS_GT].squeeze(-2).view(self.batch_size, self.dof+1)
                    trans_tar = q_tar[:, :3].view(self.batch_size, 3)
                    quat_tar = torch.cat((q_tar[:, -1].view(-1, 1), q_tar[:, 3:6]), 1).view(
                        self.batch_size, 4)
                    art_tar = q_tar[:, 6:-1].view(self.batch_size, 40)
                    
                    # compute contact labels
                    # with torch.no_grad(): 
                    #     pred_labels, _ = self.contact_label_estimation(input_rr)

                    if sim_step_idx == self.temporal_window:
                        # define q0, qdot0 before the dynamic cycle
                        q0 = q_tar.view(self.batch_size, self.dof + 1)
                        # NOTE: since we're not inferring target poses on the fly,
                        # we can obtain qdot0 from the ground truth. But I found that
                        # initializing qvel like this leads to larger motion errors.
                        # So still we initialize qdot0 to 0.
                        qdot0 = torch.zeros(
                            self.batch_size, self.dof,
                            dtype=torch.float32, device=self.device)
                        # qdot0 = data_train_per_step[
                        #     PerMotionDataName.QVEL_GT].squeeze(-2).view(self.batch_size, self.dof)
                        # if sim_step_idx == self.temporal_window:
                        #     pre_lr_th_cons = torch.zeros(
                        #         self.batch_size, 4*3,
                        #         dtype=q0.dtype, device=self.device)
                    else:
                        pass
                        # q0 = q_tar.view(self.batch_size, self.dof + 1)
                        # qdot0 = data_train_per_step[
                        #     PerMotionDataName.QVEL_GT].squeeze(-2).view(self.batch_size, self.dof)

                    # estimate inertia / forward prop
                    # NOTE: here we use our inertia estimator instead of get_mass_mat_cpu().
                    # Also, we estimate inertia for each dynamic cycle within a frame
                    if self.inertia_estimator_specs["network"] != "CRBA":
                        self.optimizer.zero_grad()
                    self.inertia_estimator.train()
                    model_input = {
                        "qpos": q0.clone().unsqueeze(1),
                        "qvel": qdot0.clone().unsqueeze(1)}
                    model_output = self.inertia_estimator(model_input)
                    
                    # extract M or M_inv
                    if self.predict_M_inv:
                        if self.inertia_estimator_specs["network"] == "CRBA":
                            M = model_output["inertia"]
                            M_inv = torch.inverse(M)
                            M_inv = ut.clean_massMat(M_inv)
                        else:
                            M_inv = model_output["inertia"]
                            M = torch.inverse(M_inv)
                    else:
                        M = model_output["inertia"]
                        M_inv = torch.inverse(M)
                        if self.inertia_estimator_specs["network"] == "CRBA":
                            M_inv = ut.clean_massMat(M_inv)
                    
                    # run the dynamic cycle
                    tau_per_cycle = []
                    for cycle_idx in range(self.num_dyn_cycles):
                        # compute ankle Jacobians
                        # J = self.cu.get_contact_jacobis6D(
                        #     self.rbdl_model,
                        #     q0.clone().cpu().detach().numpy(),
                        #     [self.rbdl_dic['left_ankle'], self.rbdl_dic['right_ankle']],
                        #     self.device)

                        # compute errors
                        quat0 = torch.cat((q0[:, -1].view(-1, 1), q0[:, 3:6]), 1)
                        errors_trans, errors_ori, errors_art = self.cu.get_PD_errors(
                            quat_tar, quat0, trans_tar, q0[:, :3], art_tar, q0[:, 6:-1])
                        current_errors = torch.cat((errors_trans, errors_ori, errors_art), 1)

                        # compute gains and tau
                        if self.neural_PD:
                            # NOTE: for DyNet's M_inv input, we use the inverse of rigid inertia
                            M_rigid = ut.get_mass_mat(
                                self.rbdl_model,
                                q0.detach().clone().cpu().numpy(),
                                device = self.device)
                            M_inv_rigid = torch.inverse(M_rigid)
                            M_inv_rigid = ut.clean_massMat(M_inv_rigid)
                            dynInput = torch.cat(
                                (q_tar, q0, qdot0, torch.flatten(M_inv_rigid, 1), current_errors,), 1)
                            neural_gain, neural_offset = self.DyNet(dynInput)
                            tau = self.cu.get_neural_development(
                                errors_trans,
                                errors_ori,
                                errors_art,
                                qdot0,
                                neural_gain,
                                neural_offset,
                                self.limit,
                                art_only = 1,
                                small_z = 1)
                        else:
                            tau = self.cu.get_tau(
                                errors_trans,
                                errors_ori,
                                errors_art,
                                qdot0,
                                self.limit,
                                small_z = 1)

                        # compute gravity and Coriolis force
                        # gcc = self.get_grav_corioli(
                        #     [0],
                        #     floor_normals,
                        #     q0,
                        #     qdot0)
                        # tau_gcc = tau + gcc
                        
                        # compute GRF/M
                        # GRFInput = torch.cat(
                        #     (tau_gcc[:, :6],
                        #      torch.flatten(J, 1),
                        #      floor_normals,
                        #      pred_labels,
                        #      pre_lr_th_cons), 1)
                        # lr_th_cons = self.GRFNet(GRFInput)
                        # gen_conF = cut.get_contact_wrench(
                        #     self.rbdl_model,
                        #     q0,
                        #     self.rbdl_dic,
                        #     lr_th_cons,
                        #     pred_labels)

                        # run Forward Dynamics and compute new q, qdot
                        # qddot = self.PyFD(tau, M_inv)
                        qddot = torch.bmm(
                            M_inv.view(self.batch_size, self.dof, self.dof),
                            tau.view(self.batch_size, self.dof, 1)).view(self.batch_size, self.dof)
                        _, q, qdot, _ = self.cu.pose_update_quat(
                            qdot0,
                            q0,
                            quat0,
                            self.delta_t,
                            qddot,
                            self.speed_limit,
                            th_zero = True,
                            disable_clamp = True)

                        # update qdot0, q0
                        qdot0 = qdot.clone()
                        q0 = self.au.angle_normalize_batch(q)

                        # register tau
                        tau_per_cycle.append(tau)
                    
                    # compute 3D joint positions (?)           
                    # p_3D_p = self.PyFK( 
                    #     [self.model_addresses["0"]],
                    #     self.target_joint_ids,
                    #     self.num_dyn_cycles * self.delta_t,
                    #     torch.FloatTensor([0]),
                    #     q0)

                    # register predictions and other stuff
                    qpos_pred[:, sim_step_idx-self.temporal_window] = q0
                    qvel_pred[:, sim_step_idx-self.temporal_window] = qdot0

                    # check per-frame pose error
                    # per_step_pose_err = torch.mean(
                    #     torch.linalg.vector_norm(
                    #     q_tar[:, 6:-1] - q0[:, 6:-1], ord=2, dim=-1),
                    #     dim=0)
                    # print(f"per-step pose error: {per_step_pose_err}")
                    
                # compute loss and backprop, after an entire sequence ends
                qpos_gt = data_train_per_batch[PerMotionDataName.QPOS_GT][
                    :, self.temporal_window:].view(
                    self.batch_size, self.seq_length-self.temporal_window, self.dof+1)
                loss_dict = self.loss_cls.loss(
                    qpos_pred,
                    qpos_gt)
                loss = loss_dict["loss"]
                if self.inertia_estimator_specs["network"] != "CRBA":
                    # qpos_pred.register_hook(
                    #     lambda grad: print(f"norm of qpos's grad: {torch.norm(grad)}"))
                    # q0.register_hook(
                    #     lambda grad: print(f"norm of qpos0's grad: {torch.norm(grad)}"))
                    # qdot0.register_hook(
                    #     lambda grad: print(f"norm of qdot0's grad: {torch.norm(grad)}"))
                    # qdot.register_hook(
                    #     lambda grad: print(f"norm of qdot's grad: {torch.norm(grad)}"))
                    # q.register_hook(
                    #     lambda grad: print(f"norm of q's grad: {torch.norm(grad)}"))
                    # qddot.register_hook(
                    #     lambda grad: print(f"norm of qddot's grad: {torch.norm(grad)}"))
                    # tau.register_hook(lambda grad: print(f"norm of tau's grad: {torch.norm(grad)}"))
                    # M.register_hook(
                    #         lambda grad: print(f"norm of M's grad: {torch.norm(grad)}"))
                    if self.predict_M_inv:
                        inertia_hook = M_inv.register_hook(
                            lambda grad: store_and_prune_grad(
                                grad, inertia_grads, True, self.grad_thresh))
                    else:
                        inertia_hook = M.register_hook(
                            lambda grad: store_and_prune_grad(
                                grad, inertia_grads, True, self.grad_thresh))
                    loss.backward()
                    inertia_hook.remove()
                    # clip grads
                    clip_grad_norm_(self.inertia_estimator.parameters(), self.grad_thresh)
                    # record grads
                    named_params = self.inertia_estimator.named_parameters()
                    for name, params in named_params:
                        assert params.grad is not None
                        if params.requires_grad:
                            weights_grads[name] = params.grad
                    self.optimizer.step()
                loss_per_step.append(loss.item())

                # visualize stuff in TB
                self.writer.add_scalar(
                    "train_loss/loss_total", loss, train_step_idx)
                self.writer.add_scalar(
                    "train_loss/loss_root_pos", loss_dict["loss_root_pos"], train_step_idx)
                self.writer.add_scalar(
                    "train_loss/loss_root_rot", loss_dict["loss_root_rot"], train_step_idx)
                self.writer.add_scalar(
                    "train_loss/loss_poses", loss_dict["loss_poses"], train_step_idx)
                if len(inertia_grads) > 0:
                    if self.predict_M_inv:
                        scalar_name = "train_grads/M_inv_grads_norm"
                    else:
                        scalar_name = "train_grads/M_grads_norm"
                    self.writer.add_scalar(
                        scalar_name,
                        torch.linalg.norm(inertia_grads[0]).cpu().detach().numpy(),
                        train_step_idx)
                for name, params_grads in weights_grads.items():
                    grad_max = torch.max(
                        torch.abs(params_grads)).cpu().detach().numpy()
                    grad_mean = torch.mean(
                        torch.abs(params_grads)).cpu().detach().numpy()
                    self.writer.add_scalar(
                        f"train_grads/{name}_grads_max",
                        grad_max,
                        train_step_idx)
                    self.writer.add_scalar(
                        f"train_grads/{name}_grads_mean",
                        grad_mean,
                        train_step_idx)
                
    def save_model(self):
        if self.inertia_estimator_specs['network'] != "CRBA":
            os.makedirs(self.save_base_path, exist_ok=True)
            nn_weights_path = f"{self.save_base_path}/{self.inertia_estimator_specs['network']}.pt"
            torch.save(
                self.inertia_estimator.state_dict(),
                nn_weights_path)
            print(f"Saved inertia estimator to {nn_weights_path}")
