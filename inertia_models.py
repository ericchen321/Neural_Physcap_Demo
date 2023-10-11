# Author: Guanxiong


import torch
import torch.nn as nn
from typing import Dict
import spdlayers
import rbdl
from Utils.misc import get_mass_mat
from Utils.angles import angle_util


def construct_symm_matrix(
    elements: torch.Tensor,
    dof: int,
    order: str = "diag") -> torch.Tensor:
    r"""
    Construct symmetric matrices from independent elements.

    Parameters:
        elements: independent elements of the symmetric matrix
            (batch_size, dof*(dof+1)/2)
        dof: degree of freedom of the system
        order: order of interpreting the elements, can be 1) "diag":
            main diag elements, then +/-1 offset diag elements, etc;
            2) "lower": lower diag elements, row-by-row; 3) "upper":
            upper diag elements, row-by-row.

    Return:
        Symmetric matrices (batch_size, dof, dof)
    """
    if order == "diag":
        elm_idxs = [0, 0]
        for offset_idx in range(0, dof):
            elm_idxs[0] = elm_idxs[1]
            elm_idxs[1] += dof - offset_idx
            if offset_idx == 0:
                # initialize diag elements
                mat_symm = torch.diag_embed(
                    elements[:, elm_idxs[0]:elm_idxs[1]],
                    offset=offset_idx)
            else:
                # assign upper and lower diag elements
                upper_diag = torch.diag_embed(
                    elements[:, elm_idxs[0]:elm_idxs[1]],
                    offset=offset_idx)
                lower_diag = upper_diag.transpose(-1, -2)
                mat_symm += upper_diag + lower_diag
    elif order == "lower":
        mat_lower = torch.zeros(
            elements.shape[0], dof, dof, device=elements.device)
        elm_idxs = [0, 0]
        for row_idx in range(dof):
            elm_idxs[0] = elm_idxs[1]
            elm_idxs[1] += row_idx + 1
            mat_lower[:, row_idx, :row_idx+1] = elements[
                :, elm_idxs[0]:elm_idxs[1]]
        mat_upper = mat_lower.transpose(-1, -2)
        # set diag elements to 0 in mat_upper
        mat_upper = mat_upper - torch.diag_embed(
            torch.diagonal(mat_upper, dim1=-2, dim2=-1))
        mat_symm = mat_lower + mat_upper
    elif order == "upper":
        raise NotImplementedError
    else:
        raise ValueError("order can only be diag, lower or upper")
    return mat_symm


def define_inertia_estimator(
    model_specs: Dict,
    seq_length: int,
    dof: int = 46,
    device: str = "cuda") -> nn.Module:
    
    # extract common NN specs
    network = model_specs["network"]
    if "activation" in model_specs:
        activation = model_specs["activation"]
    
    # decode MLP widths from depth
    if "mlp" in model_specs and "depth" in model_specs["mlp"]:
        mlp_depth = model_specs["mlp"]["depth"]
        if mlp_depth == 2:
            mlp_widths = [512]
        elif mlp_depth == 3:
            mlp_widths = [1024, 512]
        elif mlp_depth == 8:
            mlp_widths = [8192, 4096, 2048, 1024, 1024, 512, 512]
        elif mlp_depth == 12:
            mlp_widths = [4096] + \
                [2048]*2 + \
                [1024]*2 + \
                [1024]*2 + \
                [512]*2 + \
                [512]*2
        elif mlp_depth == 16:
            mlp_widths = [16384] + \
                [8192]*2 + \
                [4096]*2 + \
                [2048]*2 + \
                [1024]*2 + \
                [1024]*2 + \
                [512]*2 + \
                [512]*2
        else:
            raise ValueError("Unsupported MLP depth")
    
    # instantiate NN
    if network == "UnconNetBase":
        model = UnconNetBase(
            mlp_widths = mlp_widths,
            seq_length = seq_length,
            dof = dof,
            activation = activation).to(device)
    elif network == "SymmNetBase":
        model = SymmNetBase(
            mlp_widths = mlp_widths,
            seq_length = seq_length,
            dof = dof,
            activation = activation).to(device)
    elif network == "SPDNetBase":
        model = SPDNetBase(
            mlp_widths = mlp_widths,
            seq_length = seq_length,
            dof = dof,
            activation = activation,
            spd_layer_opts = model_specs["spd_layer"]).to(device)
    elif network == "UnconNetQVel":
        model = UnconNetQVel(
            mlp_widths = mlp_widths,
            seq_length = seq_length,
            dof = dof,
            activation = activation).to(device)
    elif network == "SymmNetQVel":
        model = SymmNetQVel(
            mlp_widths = mlp_widths,
            seq_length = seq_length,
            dof = dof,
            activation = activation).to(device)
    elif network == "SPDNetQVel":
        model = SPDNetQVel(
            mlp_widths = mlp_widths,
            seq_length = seq_length,
            dof = dof,
            activation = activation,
            spd_layer_opts = model_specs["spd_layer"]).to(device)
    elif network == "CRBA":
        model = CRBA(
            model_specs["urdf_path"])
    else:
        raise ValueError("Invalid network name")
    return model


class ArchB(nn.Module):
    def __init__(
        self,
        mlp_widths = [256, 128, 64, 32], 
        seq_length = 50,
        dof = 2,
        activation = "relu"):
        r"""
        Architecture class "B". Predict joint-space inertia from
        generalized position and the total mass of the pendulum.

        Parameters:
            mlp_widths: Width of each hidden layer, except the last
                (which has width = dof)
            seq_length: length of input sequence
            dof: Deg of freedom of the system.
            activation: activation function used in hidden layers
                can be "relu" or "lrelu"
        """
        super(ArchB, self).__init__()

        self.seq_length = seq_length
        self.dof = dof
        self.mlp_layers = nn.ModuleList([])
        for i in range(len(mlp_widths) + 1):
            if activation == "relu":
                activation_layer = nn.ReLU()
            elif activation == "lrelu":
                activation_layer = nn.LeakyReLU()
            else:
                raise ValueError(
                    "unknown activation function: {}".format(activation))
            if i == 0:
                # first hidden layer
                # takes qpos at step 0 + total sys mass
                layer = nn.Sequential(
                    nn.Linear((self.dof + 1), mlp_widths[i]),
                    activation_layer)
            elif i == len(mlp_widths):
                # last hidden layer, no activation
                layer = nn.Sequential(
                    nn.Linear(mlp_widths[i-1], self.dof**2))
            else:
                # intermediate layers
                layer = nn.Sequential(
                    nn.Linear(mlp_widths[i-1], mlp_widths[i]),
                    activation_layer)
            self.mlp_layers.append(layer)
    
    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Compute inertia from joint positions.

        Parameters:
            x: dict containing joint positions and/or velocities; each should
                be tensor (num_samples, seq_length, dof)

        Return:
            predicted integral of generalized force (num_samples, dof)
        """
        # compute inertia from joint positions
        num_sims = x["qpos"].shape[0]
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof)
        total_mass = torch.unsqueeze(
            x["link_masses"][:, 0] + x["link_masses"][:, 1],
            dim = 1).view(num_sims, 1)
        
        # just flatten. TODO: probably can do something smart here
        inertia = torch.cat(
            (qpos_init, total_mass),
            dim = 1).view(num_sims, self.dof + 1)
        for layer in self.mlp_layers:
            inertia = layer(inertia)
        inertia = inertia.view(
            (num_sims, self.dof, self.dof))
        
        # assemble output
        out = {
            "inertia": inertia}
        return out


class ArchD(ArchB):
    def __init__(
        self,
        mlp_widths = [256, 128, 64, 32], 
        seq_length = 50,
        dof = 2,
        activation = "relu"):
        r"""
        A variant of arch B: the last layer predicts the min set of
        elements necessary to construct a symmetric matrix.
        """
        super(ArchD, self).__init__(
            mlp_widths,
            seq_length,
            dof,
            activation)
        
        # compute num of elements needed
        self.num_elems = int(self.dof * (self.dof + 1) / 2)
        # replace last layer
        self.mlp_layers[-1] = nn.Linear(mlp_widths[-1], self.num_elems)

    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Compute integral of force from joint positions and velocities.
        """
        # extract initial pose
        num_sims = x["qpos"].shape[0]
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof)

        # extract total mass
        total_mass = torch.unsqueeze(
            x["link_masses"][:, 0] + x["link_masses"][:, 1],
            dim = 1).view(num_sims, 1)
        
        # augment initial pose with total mass
        inertia_params = torch.cat(
            (qpos_init, total_mass),
            dim = 1).view(num_sims, self.dof + 1)
        
        # pass through MLP
        for layer in self.mlp_layers:
            inertia_params = layer(inertia_params)
        
        # construct a symmetric matrix from the min set of elements
        inertia_params = inertia_params.view(num_sims, self.num_elems)
        inertia = construct_symm_matrix(inertia_params, self.dof)
        
        # assemble output
        out = {
            "inertia": inertia}
        return out
    

class ArchE(ArchD):
    def __init__(
        self,
        mlp_widths = [256, 128, 64, 32], 
        seq_length = 50,
        dof = 2,
        activation: str = "relu",
        spd_layer_opts: Dict = {
            "type": "cholesky",
            "min_value": 1e-8,
            "positivity": "Abs"}):
        r"""
        A variant of arch D: append an SPD layer after the MLP layers
        to guarantee that the output is an SPD matrix, or PSD matrix
        if min_value is set to 0.

        Parameters:
            activation: activation function used in MLP layers
            spd_layer_opts: Various options for the SPD layer
        """
        super(ArchE, self).__init__(
            mlp_widths,
            seq_length,
            dof,
            activation = activation)
        
        # extract SPD layer options
        spd_layer_type = spd_layer_opts["type"]
        min_value = spd_layer_opts["min_value"]
        positivity = spd_layer_opts["positivity"]
        
        # compute the input shape of the SPD layer
        self.in_shape = spdlayers.in_shape_from(dof)
        
        # define the output dim of the last MLP layer
        self.mlp_layers[-1] = nn.Linear(mlp_widths[-1], self.in_shape)

        # define the SPD layer
        if spd_layer_type == "cholesky":
            self.spd_layer = spdlayers.Cholesky(
                output_shape = dof,
                min_value = min_value,
                positive = positivity)
        elif spd_layer_type == "eigen":
            self.spd_layer = spdlayers.Eigen(
                output_shape = dof,
                min_value = min_value,
                positive = positivity)
        else:
            raise ValueError("Unsupported SPD layer type.")

    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Compute integral of force from joint positions and velocities.
        """
        # extract initial pose
        num_sims = x["qpos"].shape[0]
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof)

        # extract total mass
        total_mass = torch.unsqueeze(
            x["link_masses"][:, 0] + x["link_masses"][:, 1],
            dim = 1).view(num_sims, 1)
        
        # augment initial pose with total mass
        inertia_params = torch.cat(
            (qpos_init, total_mass),
            dim = 1).view(num_sims, self.dof + 1)
        
        # pass through MLP
        for layer in self.mlp_layers:
            inertia_params = layer(inertia_params)
        
        # pass through the SPD layer
        # NOTE: SPD layer only runs on CPU
        inertia_params = inertia_params.to("cpu")
        inertia = self.spd_layer(inertia_params)
        inertia = inertia.to(x["qpos"].device)
        
        # assemble output
        out = {
            "inertia": inertia}
        return out


class UnconNetBase(ArchB):
    def __init__(
        self,
        mlp_widths = [256, 128, 64, 32], 
        seq_length = 50,
        dof = 2,
        activation = "relu"):
        r"""
        A variant of arch B: Take no mass as input.
        """
        super(UnconNetBase, self).__init__(
            mlp_widths,
            seq_length,
            dof,
            activation)
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "lrelu":
            act_layer = nn.LeakyReLU()
        else:
            raise ValueError("Activation type not supported.")
        self.mlp_layers[0] = nn.Sequential(
            nn.Linear(self.dof+1, mlp_widths[0]),
            act_layer)

    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Compute integral of force from joint positions and velocities.

        Parameters:
            x: dict containing joint positions (num_sims, num_steps, dof+1)
        """
        # extract initial pose
        num_sims = x["qpos"].shape[0]
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof+1)

        # augment initial pose with other params
        inertia = qpos_init
        
        # pass through MLP
        for layer in self.mlp_layers:
            inertia = layer(inertia)
        inertia = inertia.view(
            (num_sims, self.dof, self.dof))
        
        # assemble output
        out = {
            "inertia": inertia}
        return out
    

class SymmNetBase(ArchD):
    def __init__(
        self,
        mlp_widths = [256, 128, 64, 32], 
        seq_length = 50,
        dof = 2,
        activation = "relu"):
        r"""
        A variant of arch D: Take no mass as input.
        """
        super(SymmNetBase, self).__init__(
            mlp_widths,
            seq_length,
            dof,
            activation)
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "lrelu":
            act_layer = nn.LeakyReLU()
        else:
            raise ValueError("Activation type not supported.")
        self.mlp_layers[0] = nn.Sequential(
            nn.Linear(self.dof+1, mlp_widths[0]),
            act_layer)

    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Compute integral of force from joint positions and velocities.

        Parameters:
            x: dict containing joint positions (num_sims, num_steps, dof+1)
        """
        # extract initial pose
        num_sims = x["qpos"].shape[0]
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof+1)
        
        # augment initial pose with other params
        inertia_params = qpos_init
        
        # pass through MLP
        for layer in self.mlp_layers:
            inertia_params = layer(inertia_params)
        
        # construct a symmetric matrix from the min set of elements
        inertia_params = inertia_params.view(num_sims, self.num_elems)
        inertia = construct_symm_matrix(inertia_params, self.dof)
        
        # assemble output
        out = {
            "inertia": inertia}
        return out


class SPDNetBase(ArchE):
    def __init__(
        self,
        mlp_widths = [256, 128, 64, 32], 
        seq_length = 50,
        dof = 2,
        activation: str = "relu",
        spd_layer_opts: Dict = {
            "type": "cholesky",
            "min_value": 1e-8,
            "positivity": "Abs"}):
        r"""
        A variant of arch E: Take no mass as input.
        """
        super(SPDNetBase, self).__init__(
            mlp_widths,
            seq_length,
            dof,
            activation,
            spd_layer_opts)
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "lrelu":
            act_layer = nn.LeakyReLU()
        else:
            raise ValueError("Activation type not supported.")
        self.mlp_layers[0] = nn.Sequential(
            nn.Linear(self.dof+1, mlp_widths[0]),
            act_layer)

    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Compute integral of force from joint positions and velocities.

        Parameters:
            x: dict containing joint positions (num_sims, num_steps, dof+1)
        """
        # extract initial pose
        num_sims = x["qpos"].shape[0]
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof+1)
        
        # augment initial pose with other params
        inertia_params = qpos_init
        
        # pass through MLP
        for layer in self.mlp_layers:
            inertia_params = layer(inertia_params)
        inertia_params_sqr = inertia_params.view(num_sims, self.in_shape)
        
        # pass through the SPD layer
        # NOTE: SPD layer only runs on CPU
        inertia_params = inertia_params.to("cpu")
        inertia = self.spd_layer(inertia_params)
        inertia = inertia.to(x["qpos"].device)
        
        # assemble output
        inertia_before_spd = construct_symm_matrix(
            inertia_params_sqr, self.dof, "lower")
        out = {
            "inertia": inertia,
            "inertia_before_spd": inertia_before_spd}
        return out


class UnconNetQVel(UnconNetBase):
    def __init__(
        self,
        mlp_widths = [256, 128, 64, 32],
        seq_length = 50,
        dof = 2,
        activation="relu"):
        r"""
        A variant of UnconNetBase: Other than a pose, take joint velocity as inputs.
        """
        super(UnconNetQVel, self).__init__(
            mlp_widths,
            seq_length,
            dof,
            activation)
        
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "lrelu":
            act_layer = nn.LeakyReLU()
        else:
            raise ValueError("Activation type not supported.")
        self.mlp_layers[0] = nn.Sequential(
            nn.Linear((2*self.dof+1), mlp_widths[0]),
            act_layer)
        
    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Compute integral of force from joint positions and velocities.

        Parameters:
            x: dict containing 1) joint positions (num_sims, num_steps, dof+1);
            2) joint velocities (num_sims, num_steps, dof)
        """
        # extract initial pose
        num_sims = x["qpos"].shape[0]
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof+1)

        # extract initial vel
        qvel_init = x["qvel"][:, 0].view(num_sims, self.dof)
        
        # augment initial pose with other params
        inertia = torch.cat(
            (qpos_init, qvel_init),
            dim = 1).view(num_sims, 2*self.dof+1)
        
        # pass through MLP
        for layer in self.mlp_layers:
            inertia = layer(inertia)
        inertia = inertia.view(
            (num_sims, self.dof, self.dof))
        
        # assemble output
        out = {
            "inertia": inertia}
        return out
    

class SymmNetQVel(SymmNetBase):
    def __init__(
        self,
        mlp_widths = [256, 128, 64, 32], 
        seq_length = 50,
        dof = 2,
        activation = "relu"):
        r"""
        A variant of SymmNetBase: Other than a pose, take joint velocity as inputs.
        """
        super(SymmNetQVel, self).__init__(
            mlp_widths,
            seq_length,
            dof,
            activation)
        
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "lrelu":
            act_layer = nn.LeakyReLU()
        else:
            raise ValueError("Activation type not supported.")
        self.mlp_layers[0] = nn.Sequential(
            nn.Linear((2*self.dof+1), mlp_widths[0]),
            act_layer)

    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Compute integral of force from joint positions and velocities.

        Parameters:
            x: dict containing 1) joint positions (num_sims, num_steps, dof+1);
            2) joint velocities (num_sims, num_steps, dof)
        """
        # extract initial pose
        num_sims = x["qpos"].shape[0]
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof+1)

        # extract initial vel
        qvel_init = x["qvel"][:, 0].view(num_sims, self.dof)
        
        # augment initial pose with other params
        inertia_params = torch.cat(
            (qpos_init, qvel_init),
            dim = 1).view(num_sims, 2*self.dof+1)
        
        # pass through MLP
        for layer in self.mlp_layers:
            inertia_params = layer(inertia_params)
        
        # construct a symmetric matrix from the min set of elements
        inertia_params = inertia_params.view(num_sims, self.num_elems)
        inertia = construct_symm_matrix(inertia_params, self.dof)
        
        # assemble output
        out = {
            "inertia": inertia}
        return out
    

class SPDNetQVel(SPDNetBase):
    def __init__(
        self,
        mlp_widths = [256, 128, 64, 32], 
        seq_length = 50,
        dof = 2,
        activation: str = "relu",
        spd_layer_opts: Dict = {
            "type": "cholesky",
            "min_value": 1e-8,
            "positivity": "Abs"}):
        r"""
        A variant of SPDNetBase: Other than a pose, take 1) joint velocity as inputs.
        """
        super(SPDNetQVel, self).__init__(
            mlp_widths,
            seq_length,
            dof,
            activation,
            spd_layer_opts)
        
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "lrelu":
            act_layer = nn.LeakyReLU()
        else:
            raise ValueError("Activation type not supported.")
        self.mlp_layers[0] = nn.Sequential(
            nn.Linear((2*self.dof+1), mlp_widths[0]),
            act_layer)
        
    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Compute integral of force from joint positions and velocities.

        Parameters:
            x: dict containing 1) joint positions (num_sims, num_steps, dof+1);
            2) joint velocities (num_sims, num_steps, dof)
        """
        # extract initial pose
        num_sims = x["qpos"].shape[0]
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof+1)

        # extract initial vel
        qvel_init = x["qvel"][:, 0].view(num_sims, self.dof)
        
        # augment initial pose with other params
        inertia_params = torch.cat(
            (qpos_init, qvel_init),
            dim = 1).view(num_sims, 2*self.dof+1)
        
        # pass through MLP
        for layer in self.mlp_layers:
            inertia_params = layer(inertia_params)
        inertia_params_sqr = inertia_params.view(num_sims, self.in_shape)
        
        # pass through the SPD layer
        # NOTE: SPD layer only runs on CPU
        inertia_params = inertia_params.to("cpu")
        inertia = self.spd_layer(inertia_params)
        inertia = inertia.to(x["qpos"].device)
        
        # assemble output
        inertia_before_spd = construct_symm_matrix(
            inertia_params_sqr, self.dof, "lower")
        out = {
            "inertia": inertia,
            "inertia_before_spd": inertia_before_spd}
        return out
    

class CRBA(nn.Module):
    r"""
    Invoke RBDL to solve mass matrices using the Composite Rigid Body
    Algorithm (CRBA).
    """
    def __init__(
        self,
        urdf_path: str,
        dof: int = 46) -> None:
        super().__init__()
        self.dof = dof
        self.model = rbdl.loadModel(urdf_path.encode(), floating_base=True)
        self.angle_util = angle_util()

    def forward(self, x: Dict[str, torch.Tensor]):
        r"""
        Parameters:
            x: Dict containing joint positions (num_sims, num_steps, dof+1)
        """
        num_sims, num_steps, _ = x["qpos"].shape
        qpos_init = x["qpos"][:, 0].view(num_sims, self.dof+1)
        M_rigid_init = get_mass_mat(
            self.model,
            qpos_init.clone().cpu().detach().numpy(),
            device = x["qpos"].device).view(num_sims, self.dof, self.dof)

        out = {
            "inertia": M_rigid_init}
        return out