import torch
import sys
import os 
import rbdl 
import warnings
import tqdm
import os 
dir_path = os.path.dirname(os.path.realpath(__file__)) 
import numpy as np
from networks import TargetPoseNetArt,TargetPoseNetOri,ContactEstimationNetwork,TransCan3Dkeys,DynamicNetwork,GRFNet
from Utils.angles import angle_util 
import Utils.misc as ut
import Utils.phys as ppf
from Utils.core_utils import CoreUtils
from Utils.initializer import InitializerConsistentHumanoid2
from lossFunctions import LossFunctions
import pybullet as p
import Utils.contacts as cut
from pipeline_util import  PyProjection,PyPerspectivieDivision
from scipy.spatial.transform import Rotation as Rot
from torch.autograd import Variable
import argparse  
from datetime import datetime
import h5py
from torch.nn.functional import mse_loss


class InferencePipeline():
    def __init__(
        self,
        urdf_path,
        net_path,
        data_path,
        save_base_path,
        w,h,K,RT,
        neural_PD=1,
        grad_descent=0,
        n_iter=6,
        con_thresh=0.01,
        limit=50,
        speed_limit=35,
        seq_name=""):

        ### configuration ###
        self.w=w
        self.h=h
        self.neural_PD=neural_PD 
        self.n_iter = n_iter
        self.con_thresh = con_thresh
        self.limit = limit
        self.speed_limit = speed_limit
        self.grad_descent = grad_descent
        self.save_base_path=save_base_path 
        self.n_iter_GD=90
        self.seq_name = seq_name

        ### joint mapping ###
        self.openpose_dic2 = { "base": 7, "left_hip": 11, "left_knee": 12, "left_ankle": 13, "left_toe": 19, "right_hip": 8, "right_knee": 9, "right_ankle": 10, "right_toe": 22, "neck": 0, "head": 14, "left_shoulder": 4, "left_elbow": 5, "left_wrist": 6, "right_shoulder": 1, "right_elbow": 2,  "right_wrist": 3 }
        self.target_joints = ["head", "neck", "left_hip", "left_knee", "left_ankle", "left_toe", "right_hip", "right_knee", "right_ankle", "right_toe", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
        self.target_ids = [self.openpose_dic2[key] for key in self.target_joints]

        ### load humanoid model ###

        self.model = rbdl.loadModel(urdf_path.encode(), floating_base=True)
        self.id_robot = p.loadURDF(urdf_path, useFixedBase=False)

        ### initilization ###
        ini = InitializerConsistentHumanoid2(n_b, self.target_joints)
        self.rbdl_dic = ini.get_rbdl_dic()
        self.target_joint_ids = ini.get_target_joint_ids()
        _, _, jointIds, jointNames = ut.get_jointIds_Names(self.id_robot)
        self.model_addresses = {"0": self.model, "1": self.model}

        ### build and load pretrained models ###
        self.TempConvNetArt = TargetPoseNetArt(in_channels=32, num_features=1024, out_channels=40, num_blocks=4)#.cuda()
        self.TempConvNetOri = TargetPoseNetOri(in_channels=32, num_features=1024, out_channels=4, num_blocks=4)#.cuda()
        self.ConNet = ContactEstimationNetwork(in_channels=32, num_features=1024, out_channels=4, num_blocks=4)#.cuda()
        self.TempConvNetTrans = TransCan3Dkeys(in_channels=32, num_features=1024, out_channels=3, num_blocks=4)#.cuda()
        self.GRFNet = GRFNet(input_dim=577, output_dim=46 + 46 + 3 * 4)#.cuda()
        self.DyNet = DynamicNetwork(input_dim=2302, output_dim=46, offset_coef=10)#.cuda()

        if os.path.exists(net_path + "ArtNet.pkl"): 
            self.TempConvNetArt.load_state_dict(torch.load(net_path + "ArtNet.pkl",  map_location=torch.device('cpu')))
            self.TempConvNetOri.load_state_dict(torch.load(net_path + "OriNet.pkl", map_location=torch.device('cpu')))
            self.ConNet.load_state_dict(torch.load(net_path + "ConNet.pkl", map_location=torch.device('cpu')))
            self.GRFNet.load_state_dict(torch.load(net_path + "GRFNet.pkl" ,map_location=torch.device('cpu')))
            self.TempConvNetTrans.load_state_dict(torch.load(net_path+"TransNet.pkl", map_location=torch.device('cpu')))
            self.DyNet.load_state_dict(torch.load(net_path+ "DyNet.pkl", map_location=torch.device('cpu'))) 
        else:
            print('no trained model found!!!')
            sys.exit()
        self.TempConvNetArt.eval()
        self.TempConvNetOri.eval()
        self.TempConvNetTrans.eval()
        self.ConNet.eval()
        self.DyNet.eval()
        self.GRFNet.eval()

        ### setup custom pytorch functions including the Physics model 
        self.PyFK = ppf.PyForwardKinematicsQuaternion().apply 
        self.PyFK_rr = ppf.PyForwardKinematicsQuaternion().apply 
        self.PyFD = ppf.PyForwardDynamics.apply
        self.PyProj =  PyProjection.apply
        self.PyPD =   PyPerspectivieDivision.apply

        ### load input data ###
        self.RT=RT
        self.Rs = torch.FloatTensor(self.RT[:3, :3]).view(n_b, 3, 3)
        self.P = torch.FloatTensor(K[:3])
        self.P_tensor = self.get_P_tensor(n_b, self.target_joint_ids, self.P)
        self.p_2ds = np.load(data_path)

        self.p_2d_basee = self.p_2ds[:, self.openpose_dic2["base"]]
        self.p_2ds = self.p_2ds[:, self.target_ids]
        self.p_2ds = torch.FloatTensor(self.p_2ds)
        self.p_2d_basee = torch.FloatTensor(self.p_2d_basee)

        self.canoical_2Ds = self.canonicalize_2Ds(torch.FloatTensor(K[:3, :3]), self.p_2ds)
        self.p_2ds[:, :, 0] /= self.w
        self.p_2ds[:, :, 1] /= self.h  # h
        self.p_2d_basee[:, 0] /= self.w
        self.p_2d_basee[:, 1] /= self.h  # h
        self.p_2ds_rr = self.p_2ds - self.p_2d_basee.view(-1, 1, 2)

    def get_P_tensor(self,N, target_joint_ids, P):
        P_tensor = torch.zeros(N, 3 * len(target_joint_ids), 4 * len(target_joint_ids))
        for i in range(int(P_tensor.shape[1] / 3)):
            P_tensor[:, i * 3:(i + 1) * 3, i * 4:(i + 1) * 4] = P
        return torch.FloatTensor(np.array(P_tensor))

    def canonicalize_2Ds(self,K, p_2Ds):
        cs = torch.FloatTensor([K[0][2], K[1][2]]).view(1, 1, 2)
        fs = torch.FloatTensor([K[0][0], K[1][1]]).view(1, 1, 2)
        canoical_2Ds = (p_2Ds - cs) / fs
        return canoical_2Ds

    def get_grav_corioli(self,sub_ids, floor_noramls, q, qdot):
        n_b, _ = q.shape
        q = q.cpu().numpy().astype(float)
        qdot = qdot.cpu().numpy().astype(float)
        gcc = np.zeros((n_b, self.model.qdot_size))
        floor_noramls = floor_noramls.cpu().numpy()
        for batch_id in range(n_b):
            sid = sub_ids[batch_id]
            model_address = self.model_addresses[str(int(sid))]
            model_address.gravity = -9.8 * floor_noramls[batch_id]
            rbdl.InverseDynamics(model_address, q[batch_id], qdot[batch_id], np.zeros(self.model.qdot_size).astype(float),  gcc[batch_id]) 
        return torch.FloatTensor(gcc)

    def get_mass_mat(self,model, q):
        n_b, _ = q.shape
        M_np = np.zeros((n_b, model.qdot_size, model.qdot_size))
        [rbdl.CompositeRigidBodyAlgorithm(model, q[i].astype(float), M_np[i]) for i in range(n_b)]
        return M_np
 

    def contact_label_estimation(self,input_rr):
        pred_labels = self.ConNet(input_rr)
        pred_labels = pred_labels.clone().cpu()
        pred_labels_prob = pred_labels.clone()
        pred_labels[pred_labels < self.con_thresh] = 0
        pred_labels[pred_labels >= self.con_thresh] = 1
        return pred_labels,pred_labels_prob

    def gradientDescent(self,trans0,target_2D,rr_3ds):
        trans_variable = trans0.clone()
        for j in range(self.n_iter_GD):
            trans_variable = Variable(trans_variable, requires_grad=True)
            p_3D =(rr_3ds.view(n_b,-1,3)+trans_variable.view(n_b,1,3)).view(n_b,-1)

            p_proj = self.PyProj(self.P_tensor, p_3D)
            p_2D = self.PyPD(p_proj)
            p_2D = p_2D.view(n_b,-1,2)
            p_2D[:,:,0]/=self.w
            p_2D[:,:,1]/=self.h
            loss2D = (p_2D.view(1,  -1) - target_2D.view(1,  -1)).pow(2).sum() + 10* (trans_variable-trans0).pow(2).sum()
             
            loss2D.backward()
            with torch.no_grad():
                trans_variable -= 0.003  * trans_variable.grad
                trans_variable.grad.zero_()

            trans_variable = trans_variable.clone().detach()
            p_2D = p_2D.detach() 

        p_2D = p_2D.clone().detach()#*1000
        p_2D = p_2D.view(1, -1, 2)
        #p_2D[:, :, 0] *= self.w
        #p_2D[:, :, 1] *=self.h
        return trans_variable ,p_2D

    def get_translations_GD(self,target_2D,rr_p_3D_p,trans0):
         

        """ set 2D and 3D targets """ 
        trans, _ = self.gradientDescent( trans0, target_2D, rr_p_3D_p.view(n_b, -1))

        #target_2D = target_2D.view(n_b, -1, 2)
        #target_2D[:, :, 0] *= self.w
        #target_2D[:, :, 1] *= self.h
 
        return trans.clone()

    def get_target_pose(self,input_can,input_rr,target_2d,trans0,first_frame_flag):
        art_tar = self.TempConvNetArt(input_rr)
        quat_tar = self.TempConvNetOri(input_rr)
        rr_q = torch.cat((torch.zeros(n_b, 3) , quat_tar[:, 1:], art_tar, quat_tar[:, 0].view(-1, 1)), 1)#.cuda()
        
        rr_p_3D_p = self.PyFK_rr([self.model_addresses["0"]], self.target_joint_ids,delta_t, torch.FloatTensor([0]) , rr_q) 
        q_tar = rr_q.clone()

        if not first_frame_flag and self.grad_descent:
            trans_tar = self.get_translations_GD(target_2d.cpu(),rr_p_3D_p.cpu().detach(),trans0.cpu().detach())
        else: 
            trans_tar = self.TempConvNetTrans(input_can, rr_p_3D_p)
            trans_tar = torch.clamp(trans_tar, -50, 50)

        q_tar[:, :3] = trans_tar.clone()
        return art_tar,quat_tar,trans_tar,q_tar

    def inference(self):

        ### Initialization ###
        qpos_opt , p_3ds_opt, tau_special_iters, qpos_opt_iters =  [],[],[],[]
        tau_opt_iters = []
        qvel_opt_iters = []
        bfrc_gr_opt_iters = []
        qfrc_gr_opt_iters = []
        gravcol_iters = []
        M_rigid_iters = []
        qpos_gt = []
        qvel_gt = []
        qvel_opt = []
        bfrc_gr_opt = []
        qfrc_gr_opt = []
        M_rigid = []
        tau_opt = []
        gravcol = []
      
        p_2ds_rr = self.p_2ds_rr#.cuda()
        canoical_2Ds = self.canoical_2Ds#.cuda()
        p_2ds = self.p_2ds#.cuda()  
        ### set axis vectors ###
        basis_vec_w = torch.FloatTensor(np.array([[1, 0, 0, ], [0, 1, 0, ], [0, 0, 1, ]])).view(1, 3, 3)
        basis_vec_w = basis_vec_w.expand(n_b, -1, -1)
        
        # print(p_2ds_rr.shape)
        # while True:
        #     pass
        for i in tqdm.tqdm(list(range(temporal_window, len(p_2ds_rr)))):
            frame_canonical_2Ds = canoical_2Ds[i - temporal_window:i, ].reshape(n_b, temporal_window, -1)
            frame_rr_2Ds = p_2ds_rr[i - temporal_window:i, ].reshape(n_b, temporal_window, -1)
            floor_noramls = torch.transpose(torch.bmm(self.Rs, torch.transpose(basis_vec_w, 1, 2)), 1, 2)[:, 1].view(n_b, 3)
            input_rr = frame_rr_2Ds.reshape(n_b, temporal_window, -1)
            input_can = frame_canonical_2Ds.reshape(n_b, temporal_window, -1)
            target_2d = p_2ds[i].reshape(n_b,-1)

            if i==temporal_window:
                tar_trans0 = None 
                first_frame_flag=1
            else:
                first_frame_flag=0

            ### compute Target Pose ###
            art_tar, quat_tar, trans_tar, q_tar = self.get_target_pose(input_can,input_rr,target_2d,tar_trans0,first_frame_flag)
            # print(f"quat_tar.shape: {quat_tar.shape}")
            # print(f"q_tar.shape: {q_tar.shape}")
            # while True:
            #     pass
            tar_trans0=trans_tar.clone()
            with torch.no_grad(): 
                ### compute contact labels ###
                pred_labels,pred_labels_prob = self.contact_label_estimation(input_rr)

                if i == temporal_window:
                    q0 = q_tar.clone().cpu()
                    pre_lr_th_cons = torch.zeros(n_b, 4 * 3)
                    # print(f"qdot_size : {self.model.qdot_size}")
                    # while True:
                    #     pass
                    qdot0 = torch.zeros(n_b, self.model.qdot_size)

                quat_tar = quat_tar.cpu()
                art_tar = art_tar.cpu()
                trans_tar = trans_tar.cpu()
                q_tar = q_tar.cpu()

                ### Dynamic Cycle ###
                for iter in range(self.n_iter):
                    ### Compute dynamic quantitites and pose errors ###
                    M = ut.get_mass_mat_cpu(self.model, q0.detach().clone().cpu().numpy())
                    # print(f"cond num of M: {np.linalg.cond(M)}")
                    M_inv = torch.inverse(M).clone()
                    # print(f"norm of M_inv before cleaning: {torch.norm(M_inv)}")
                    M_inv = ut.clean_massMat(M_inv)
                    # print(f"norm of M_inv after cleaning: {torch.norm(M_inv)}")
                    J = CU.get_contact_jacobis6D_cpu(self.model, q0.numpy(), [self.rbdl_dic['left_ankle'], self.rbdl_dic['right_ankle']])  # ankles

                    quat0 = torch.cat((q0[:, -1].view(-1, 1), q0[:, 3:6]), 1).detach().clone()
                    errors_trans, errors_ori, errors_art = CU.get_PD_errors_cpu(quat_tar, quat0, trans_tar, q0[:, :3], art_tar, q0[:, 6:-1])
                    current_errors = torch.cat((errors_trans, errors_ori, errors_art), 1)

                    ### Force Vector Computation ###
                    if self.neural_PD:
                        # print(f"qdot0's norm: {torch.norm(qdot0)}")
                        # print(f"qdot0.shape: {qdot0.shape}")
                        # while True:
                        #     pass
                        dynInput = torch.cat((q_tar, q0, qdot0, torch.flatten(M_inv, 1), current_errors,), 1)
                        neural_gain, neural_offset = self.DyNet(dynInput )#.cuda()
                        tau = CU.get_neural_development_cpu(errors_trans, errors_ori, errors_art, qdot0, neural_gain.cpu(), neural_offset.cpu(), self.limit,art_only=1, small_z=1)
                    else:
                        tau = CU.get_tau(errors_trans, errors_ori, errors_art, qdot0, self.limit, small_z=1)

                    gcc = self.get_grav_corioli([0], floor_noramls, q0.clone(), qdot0.clone())
                    tau_gcc = tau + gcc

                    ### GRF computation ###
                    GRFInput = torch.cat((tau_gcc[:, :6], torch.flatten(J, 1), floor_noramls, pred_labels, pre_lr_th_cons), 1)#.cuda()
                    lr_th_cons = self.GRFNet(GRFInput)
                    gen_conF = cut.get_contact_wrench_cpu(self.model, q0, self.rbdl_dic, lr_th_cons.cpu(), pred_labels)
                    # print(f"lr_th_cons.shape: {lr_th_cons.shape}")
                    # print(f"gen_conF.shape: {gen_conF.shape}")
                    # while True:
                    #     pass
                     
                    ### Forward Dynamics and Pose Update ###
                    tau_special = tau_gcc - gen_conF
                    qfrc_net = tau_special + gen_conF - gcc
                    qddot = self.PyFD(qfrc_net, M_inv)
                    quat, q, qdot, _ = CU.pose_update_quat_cpu(qdot0.detach(), q0.detach(), quat0.detach(), delta_t, qddot, self.speed_limit, th_zero=1)
                    
                    # inspect per-iteration impulse loss
                    # Mdv_iter = torch.bmm(
                    #     torch.FloatTensor(M),
                    #     torch.FloatTensor(qdot - qdot0).view(n_b, 46, 1))
                    # impl_opt_iter = delta_t * torch.FloatTensor(qfrc_net)
                    # loss_iter = mse_loss(Mdv_iter, impl_opt_iter)
                    # print(f"impulse loss (in cycle): {loss_iter}")

                    # inspect per-iteration velocity loss
                    # qvel_diff_iter = torch.FloatTensor(qdot - qdot0).view(n_b, 46, 1)
                    # Minv_tau_dt_iter = delta_t * torch.bmm(
                    #     torch.FloatTensor(M_inv),
                    #     torch.FloatTensor(qfrc_net).view(n_b, 46, 1))
                    # loss_iter = mse_loss(qvel_diff_iter, Minv_tau_dt_iter)
                    # print(f"qvel_diff_iter's norm: {torch.norm(qvel_diff_iter)}")
                    # print(f"Minv_tau_dt_iter's norm: {torch.norm(Minv_tau_dt_iter)}")
                    # print(f"velocity loss (in cycle): {loss_iter}")

                    # NOTE: here we sample per-iteration data
                    # if iter == 0:
                    #     tau_special_iters.append(torch.flatten(tau_special).numpy())
                    tau_special_iters.append(tau_special.numpy())
                    qpos_opt_iters.append(q0.cpu().detach().numpy())
                    qvel_opt_iters.append(qdot0.numpy())
                    bfrc_gr_opt_iters.append(lr_th_cons.cpu().detach().numpy())
                    qfrc_gr_opt_iters.append(gen_conF.numpy())
                    tau_opt_iters.append(qfrc_net.cpu().detach().numpy())
                    gravcol_iters.append(gcc.cpu().detach().numpy())
                    M_rigid_iters.append(M.cpu().detach().numpy())
                    
                    # update kinematic state
                    qdot0 = qdot.detach().clone()
                    q0 = AU.angle_normalize_batch_cpu(q.detach().clone())
                
                ### store the predictions ###             
                p_3D_p = self.PyFK( [self.model_addresses["0"]], self.target_joint_ids,delta_t, torch.FloatTensor([0]) , q0) 
                qpos_opt.append(q0.cpu().detach().numpy()) 
                p_3ds_opt.append(p_3D_p[0].cpu().numpy()) 

                # store kin + dyn data
                qpos_gt.append(q_tar.detach().numpy())
                if len(qpos_gt) >= 2:
                    qvel_gt.append(
                        AU.differentiate_qpos(
                            qpos_gt[-1], qpos_gt[-2], self.n_iter*delta_t))
                qvel_opt.append(qdot.detach().numpy())
                M_rigid.append(M.detach().numpy())
                # NOTE: here we compute the averaged forces over the last dynamic cycle
                bfrc_gr_opt.append(
                    np.mean(np.array(
                        bfrc_gr_opt_iters[-1*self.n_iter:]).reshape(self.n_iter, n_b, 12),
                    axis=0))
                qfrc_gr_opt.append(
                    np.mean(np.array(
                        qfrc_gr_opt_iters[-1*self.n_iter:]).reshape(self.n_iter, n_b, 46),
                    axis=0))
                tau_opt.append(
                    np.mean(np.array(
                        tau_opt_iters[-1*self.n_iter:]).reshape(self.n_iter, n_b, 46),
                    axis=0))
                gravcol.append(
                    np.mean(np.array(
                        gravcol_iters[-1*self.n_iter:]).reshape(self.n_iter, n_b, 46),
                    axis=0))

                # WTS impulse loss is large
                # q_tar_norm = AU.angle_normalize_batch_cpu(q_tar.detach().clone())
                # print(f"||norm(q_tar) - q||: {torch.norm(q_tar_norm - q0)}"
                # if len(qvel_gt) >= 2:
                #     Mdv = torch.bmm(
                #         torch.FloatTensor(M),
                #         torch.FloatTensor(qvel_gt[-1] - qvel_gt[-2]).view(n_b, 46, 1))
                #     impl = self.n_iter * delta_t * torch.FloatTensor(tau_opt[-1]).view(n_b, 46, 1)
                #     # qfrc_nets = torch.FloatTensor(np.array(
                #     #     tau_opt_iters[-1*self.n_iter:])).view(self.n_iter, n_b, 46, 1)
                #     # impl = self.n_iter * delta_t * torch.sum(qfrc_nets, 0)
                #     loss = mse_loss(Mdv, impl)
                #     print(f"impulse loss (out of cycle) using GT qvel: {loss}")
                # if len(qvel_opt) >= 2:
                #     Mdv = torch.bmm(
                #         torch.FloatTensor(M),
                #         torch.FloatTensor(qvel_opt[-1] - qvel_opt[-2]).view(n_b, 46, 1))
                #     qfrc_nets = torch.FloatTensor(np.array(
                #             tau_opt_iters[-1*self.n_iter:])).view(self.n_iter, n_b, 46, 1)
                #     impl = self.n_iter * delta_t * torch.sum(qfrc_nets, 0)
                #     loss = mse_loss(Mdv, impl)
                #     print(f"impulse loss (out of cycle) using opt qvel: {loss}")

                # WTS velocity loss is large
                # if len(qvel_gt) >= 2:
                #     qvel_diff_gt = torch.FloatTensor(qvel_gt[-1] - qvel_gt[-2]).view(n_b, 46, 1)
                #     qvel_diff_pred = self.n_iter * delta_t * torch.bmm(
                #         torch.FloatTensor(M_inv),
                #         torch.FloatTensor(tau_opt[-1]).view(n_b, 46, 1))
                #     # qfrc_nets = torch.FloatTensor(np.array(
                #     #     tau_opt_iters[-1*self.n_iter:])).view(self.n_iter, n_b, 46, 1)
                #     # impl = self.n_iter * delta_t * torch.sum(qfrc_nets, 0)
                #     # qvel_diff_pred = torch.bmm(
                #     #     torch.FloatTensor(M_inv),
                #     #     impl.view(n_b, 46, 1))
                #     print(f"norm of M: {torch.norm(M)}")
                #     print(f"norm of M_inv: {torch.norm(M_inv)}")
                #     print(f"norm of qvel_diff_pred: {torch.norm(qvel_diff_pred)}")
                #     loss = mse_loss(qvel_diff_pred, qvel_diff_gt)
                #     print(f"velocity loss (out of cycle) using GT qvel: {loss}")
                # if len(qvel_opt) >= 2:
                #     qvel_diff_opt = torch.FloatTensor(qvel_opt[-1] - qvel_opt[-2]).view(n_b, 46, 1)
                #     qvel_diff_pred = self.n_iter * delta_t * torch.bmm(
                #         torch.FloatTensor(M_inv), torch.FloatTensor(tau_opt[-1]).view(n_b, 46, 1))
                #     loss = mse_loss(qvel_diff_pred, qvel_diff_opt)
                #     print(f"velocity loss using opt qvel: {loss}")

                # check joint pose error
                # print(f"||q_tar - q||: {torch.norm(q_tar - q0, 1)}")
         
        ########### save the predictions ###############
        print('saving predictions ...')
        qpos_opt_flat = np.array(qpos_opt).reshape(len(qpos_opt), -1)
        p_3ds_opt = np.array(p_3ds_opt) 
        qpos_opt_iters_flat = np.array(qpos_opt_iters).reshape(len(qpos_opt_iters), -1)
        if not os.path.exists(self.save_base_path +  "/"): 
            os.makedirs(self.save_base_path + "/")
        np.save(self.save_base_path +  "/qpos_opt_iters_flat.npy", qpos_opt_iters_flat)
        np.save(self.save_base_path +  "/qpos_opt_flat.npy", qpos_opt_flat)   
        np.save(self.save_base_path +  "/p_3ds_opt.npy", p_3ds_opt)  
        
        # save all kin + dyn data in h5 format
        print('saving all kin + dyn data in h5 file...')
        qpos_gt = np.array(qpos_gt[1:])[:, 0]
        qvel_gt = np.array(qvel_gt)[:, 0]
        bfrc_gr_opt = np.array(bfrc_gr_opt[1:])[:, 0]
        qfrc_gr_opt = np.array(qfrc_gr_opt[1:])[:, 0]
        M_rigid = np.array(M_rigid[1:])[:, 0]
        tau_opt = np.array(tau_opt[1:])[:, 0]
        gravcol = np.array(gravcol[1:])[:, 0]
        qpos_opt_iters = np.array(qpos_opt_iters)[:, 0]
        qvel_opt_iters = np.array(qvel_opt_iters)[:, 0]
        bfrc_gr_opt_iters = np.array(bfrc_gr_opt_iters)[:, 0]
        qfrc_gr_opt_iters = np.array(qfrc_gr_opt_iters)[:, 0]
        tau_opt_iters = np.array(tau_opt_iters)[:, 0]
        gravcol_iters = np.array(gravcol_iters)[:, 0]
        M_rigid_iters = np.array(M_rigid_iters)[:, 0]
        print(f"qpos_gt.shape: {qpos_gt.shape}")
        print(f"qvel_gt.shape: {qvel_gt.shape}")
        print(f"bfrc_gr_opt.shape: {bfrc_gr_opt.shape}")
        print(f"qfrc_gr_opt.shape: {qfrc_gr_opt.shape}")
        print(f"tau_opt.shape: {tau_opt.shape}")
        print(f"gravcol.shape: {gravcol.shape}")
        print(f"M_rigid.shape: {M_rigid.shape}")
        print(f"qpos_opt_iters.shape: {qpos_opt_iters.shape}")
        print(f"qvel_opt_iters.shape: {qvel_opt_iters.shape}")
        print(f"bfrc_gr_opt_iters.shape: {bfrc_gr_opt_iters.shape}")
        print(f"qfrc_gr_opt_iters.shape: {qfrc_gr_opt_iters.shape}")
        print(f"tau_opt_iters.shape: {tau_opt_iters.shape}")
        print(f"gravcol_iters.shape: {gravcol_iters.shape}")
        print(f"M_rigid_iters.shape: {M_rigid_iters.shape}")
        h5path = f"{self.save_base_path}/data+seq_name={self.seq_name}.h5"
        with h5py.File(h5path, "w") as h5file:
            # save data
            h5file.create_dataset(
                "qpos_gt", data = qpos_gt)
            h5file.create_dataset(
                "qvel_gt", data = qvel_gt)
            h5file.create_dataset(
                "bfrc_gr_opt", data = bfrc_gr_opt)
            h5file.create_dataset(
                "qfrc_gr_opt", data = qfrc_gr_opt)
            h5file.create_dataset(
                "tau_opt", data = tau_opt)
            h5file.create_dataset(
                "gravcol", data = gravcol)
            h5file.create_dataset(
                "M_rigid", data = M_rigid)
            h5file.create_dataset(
                "qpos_opt_iters", data = qpos_opt_iters)
            h5file.create_dataset(
                "qvel_opt_iters", data = qvel_opt_iters)
            h5file.create_dataset(
                "bfrc_gr_opt_iters", data = bfrc_gr_opt_iters)
            h5file.create_dataset(
                "qfrc_gr_opt_iters", data = qfrc_gr_opt_iters)
            h5file.create_dataset(
                "tau_opt_iters", data = tau_opt_iters)
            h5file.create_dataset(
                "gravcol_iters", data = gravcol_iters)
            h5file.create_dataset(
                "M_rigid_iters", data = M_rigid_iters)
        print(f"Kin + dyn data saved to {h5path}")
        print('Done.')
        return 0
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='arguments for predictions')
    parser.add_argument('--input_path', default='sample_data/sample_dance.npy')
    parser.add_argument('--net_path', default="../pretrained_neuralPhys/")
    parser.add_argument('--n_iter', type=int, default=6)
    parser.add_argument('--con_thresh', type=float, default= 0.001 )
    parser.add_argument('--tau_limit', type=float, default= 80 )
    parser.add_argument('--speed_limit', type=float, default=15) 
    parser.add_argument('--img_width', type=float, default=1280)
    parser.add_argument('--img_height', type=float, default=720)
    parser.add_argument('--floor_known', type=int, default=1)
    parser.add_argument('--floor_position_path', default="./sample_data/sample_floor_position.npy")
    parser.add_argument('--cam_params_known', type=int, default=0)
    parser.add_argument('--cam_params_path', default="./sample_data/sample_cam_params.npy")
    parser.add_argument('--save_base_path', default="./data_logging/")
    parser.add_argument('--urdf_path', default="./URDF/manual.urdf")
    parser.add_argument('--temporal_window', type=int, default=10)
    args = parser.parse_args()

    """
    todo:
    start from frame 1 (not 10)
    """
    AU = angle_util()
    LF = LossFunctions()
    delta_t = 0.011
    CU = CoreUtils(45, delta_t)
    warnings.filterwarnings("ignore")
    n_b = 1 
    id_simulator = p.connect(p.DIRECT)
    p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1) 
    time_curr = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    input_basename = os.path.splitext(
        os.path.basename(args.input_path))[0]
    save_base_path = "".join(
        [f"{args.save_base_path}/",
         "demo/",
         f"{input_basename}+t={time_curr}"])
    urdf_path = args.urdf_path
    net_path=args.net_path  
    w=args.img_width
    h=args.img_height
    temporal_window = args.temporal_window
 
    if args.floor_known: 
        RT = np.load(args.floor_position_path ) 
    else:
        RT = None 

    if args.cam_params_known:  
        K = np.load(args.cam_params_path) 
        grad_descent=0
    else: 
        K = np.array([1000, 0, w/2, 0, 0, 1000, h/2, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4) 
        grad_descent=1
    
    IPL = InferencePipeline(
        urdf_path,
        net_path, 
        args.input_path,
        save_base_path,
        w,h,K,RT,
        neural_PD=1,
        grad_descent=grad_descent, 
        n_iter=args.n_iter,
        con_thresh=args.con_thresh,
        limit=args.tau_limit,
        speed_limit=args.speed_limit,
        seq_name=input_basename)
    IPL.inference()