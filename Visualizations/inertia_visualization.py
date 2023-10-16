# Author: Guanxiong


from typing import List, Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os


def store_grad(grad, storing_list):
    r"""
    Hook function for storing gradient tensor in a list. Should be
    used with register_hook().
    """
    storing_list.append(grad.clone())


class NPhysCapDoFSemantics:
    r"""
    What each DoF in the dataset stands for.
    """

    def __init__(self) -> None:
        self.dof = 46
        self.dof_semantics = []
        for i in range(self.dof):
            self.dof_semantics.append(f"{i:02d}")


class MatrixVisualizer:
    r"""
    Visualize matrices (e.g. inertia, stiffness) of motion sequences.
    """

    def __init__(self, matrix_name: str):
        self.matrix_name = matrix_name

    def plot_cond_numbers_vs_time(
        self,
        network_names: List[str],
        times: Union[List, np.ndarray],
        matrices: np.ndarray,
        plot_dir: str):
        r"""
        Plot condition number of mutliple matrices vs time, and save the plot to disk.

        Parameters:
            network_names: Names of networks, e.g. ["baseline", "model of interest"]
            times: List of times, in ms, at which data are sampled (num_steps,)
            matrices: Matrix of each network (num_networks, num_steps, dim, dim)
            plot_dir: Directory to save plots to
        """
        # precondition checks
        num_networks, num_steps, dim, _ = matrices.shape
        assert num_networks == len(network_names)
        assert num_steps == len(times)

        # create directory if it doesn't exist
        save_dir = f"{plot_dir}/{self.matrix_name}_cond_numbers"
        os.makedirs(save_dir, exist_ok=True)

        # compute condition numbers
        conds = np.zeros((num_networks, num_steps))
        for nw_idx in range(num_networks):
            for step_idx in range(num_steps):
                conds[nw_idx, step_idx] = np.linalg.cond(matrices[nw_idx, step_idx])

        # set up figure dimensions
        matplotlib.use('Agg')
        matplotlib.rcParams['figure.dpi'] = 100
        dpi = matplotlib.rcParams['figure.dpi']
        num_rows = num_networks
        num_cols = 1
        fig_w, fig_h = (500*num_cols, 500*num_rows)
        figsize = fig_w / float(dpi), fig_h / float(dpi)

        # plot condition numbers
        fig, axs = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            sharex="col",
            sharey="none",
            squeeze=False,
            figsize=figsize)
        assert num_steps == len(times)
        for row_idx in range(num_rows):
            axs[row_idx][0].plot(
                times, conds[row_idx, :],
                label=network_names[row_idx])
            axs[row_idx][0].legend()
        plt.xlabel("Time (ms)")
        plt.ylabel("Condition Number")
        plt.tight_layout()
        img_path = f"{save_dir}/{self.matrix_name}_cond_numbers_vs_time.png"
        plt.savefig(img_path)
        plt.close(fig)
        print(f"Condition number plot saved to {img_path}")



class InertiaMatrixVisualizer(MatrixVisualizer):
    r"""
    Visualize inertia matrices of motion sequences.
    """

    def __init__(
        self,
        matrix_name: str = "inertia"):
        super().__init__(matrix_name)

        # semantic meaning of each DoF
        self.dof_semantics = NPhysCapDoFSemantics().dof_semantics
        for dof_idx in range(len(self.dof_semantics)):
            self.dof_semantics[dof_idx] = f"{dof_idx}: {self.dof_semantics[dof_idx]}"

    def plot_inertia_as_heatmaps(
        self,
        network_names: List[str],
        times: Union[List, np.ndarray],
        Ms: np.ndarray,
        plot_dir: str,
        plot_first_step_only: bool = False):
        r"""
        Plot per-step estimated inertia of multiple networks.

        Parameters:
            network_names: Names of networks, e.g. ["baseline", "model of interest"].
                Need at least two networks.
            times: List of times, in ms, at which data are sampled (num_steps,)
            Ms: Inertia matrix of each network
                (num_networks, num_steps, dim, dim)
            plot_dir: Directory to save plots to
            plot_first_step_only: Whether to plot only the first step
        """
        # precondition checks
        num_networks, num_steps, dim, _ = Ms.shape
        assert num_networks == len(network_names), \
            f"num_networks: {num_networks}, len(network_names): {len(network_names)}"
        assert num_steps == len(times), \
            f"num_steps: {num_steps}, len(times): {len(times)}"

        # create directory if it doesn't exist
        save_dir = f"{plot_dir}/{self.matrix_name}_heatmaps"
        os.makedirs(save_dir, exist_ok=True)

        # compute diagonally-lumped inertia
        M_diags = np.sum(Ms, axis=2)
        Ms_dlumped = np.zeros((num_networks, num_steps, dim, dim))
        for nw_idx in range(num_networks):
            for step_idx in range(num_steps):
                Ms_dlumped[nw_idx, step_idx, :, :] = np.diag(M_diags[nw_idx, step_idx, :])

        # set up figure dimensions
        matplotlib.use('Agg')
        matplotlib.rcParams['figure.dpi'] = 100
        dpi = matplotlib.rcParams['figure.dpi']
        num_rows = 2
        num_cols = num_networks + 1
        fig_w, fig_h = (700*num_cols+300, 700*num_rows)
        figsize = fig_w / float(dpi), fig_h / float(dpi)
        width_ratios = [8.5] + (num_cols-2)*[8] + [0.25]
        
        for step_idx in range(num_steps):
            fig, axs = plt.subplots(
                nrows=num_rows, ncols=num_cols,
                figsize=figsize,
                squeeze=False,
                width_ratios=width_ratios)
            
            # compute anchoring values for colorbars. Do this separately
            # for "raw" and diagonally-lumped mass matrices
            val_min_Ms = np.amin(Ms[:, step_idx])
            val_max_Ms = np.amax(Ms[:, step_idx])
            val_min_Ms_dlumped = np.amin(Ms_dlumped[:, step_idx])
            val_max_Ms_dlumped = np.amax(Ms_dlumped[:, step_idx])
                
            for nw_idx in range(num_networks):
                for row_idx in range(num_rows):
                    # pick center value + colormap for the colorbar
                    if row_idx == 0:
                        val_min = val_min_Ms
                        val_max = val_max_Ms
                    else:
                        val_min = val_min_Ms_dlumped
                        val_max = val_max_Ms_dlumped
                    if val_min < 0 and val_max > 0:
                        # set center of colorbar to 0
                        center = 0
                        cmap = "seismic"
                    elif val_min >= 0:
                        # set bottom of colorbar to 0
                        center = None
                        cmap = "Blues"
                    else:
                        raise ValueError("val_max should be > 0")
                    
                    # get inertia / diagonally-lumped inertia
                    if row_idx == 0:
                        matrix = Ms[nw_idx, step_idx]
                    else:
                        matrix = Ms_dlumped[nw_idx, step_idx]

                    if nw_idx == 0:
                        # plot the 1st network's inertia + dof semantic labels +
                        # colorbar (in the last column)
                        sns.heatmap(
                            matrix,
                            linewidth=0.5, linecolor="black", ax=axs[row_idx][0],
                            square=True,
                            vmin=val_min, vmax=val_max, center=center, cmap=cmap,
                            xticklabels=False, yticklabels=self.dof_semantics,
                            cbar_ax = axs[row_idx][-1])
                    else:
                        # plot other network's estimated inertia
                        sns.heatmap(
                            matrix,
                            linewidth=0.5, linecolor="black", ax=axs[row_idx][nw_idx],
                            square=True,
                            vmin=val_min, vmax=val_max, center=center, cmap=cmap,
                            xticklabels=list(range(dim)), yticklabels=list(range(dim)),
                            cbar=False)                    

                    # set subplot title
                    if row_idx == 0: 
                        axs[row_idx][nw_idx].set_title(network_names[nw_idx])
                    else:
                        axs[row_idx][nw_idx].set_title(
                            f"{network_names[nw_idx]},\ndiagonally-lumped")
            
            # save the plot
            img_path = f"{save_dir}/{self.matrix_name}_heatmaps+time={times[step_idx]:03d}ms.png"
            plt.savefig(img_path)
            plt.close(fig)

            # if plotting only the first step, break
            if plot_first_step_only:
                break

        if plot_first_step_only:
            print(f"Inertia heatmap plots of the first step saved to {plot_dir}")
        else:
            print(f"Inertia heatmap plots of {num_steps} steps saved to {plot_dir}")

    def plot_eigenvals_of_inertia(
        self,
        network_names: List[str],
        times: Union[List, np.ndarray],
        matrices: np.ndarray,
        plot_dir: str,
        plot_first_step_only: bool = False):
        r"""
        Plot per-step eigenvalues of estimated inertia of multiple networks.

        Parameters:
            network_names: Names of networks, e.g. ["baseline", "model of interest"].
                Need at least two networks.
            times: List of times, in ms, at which data are sampled (num_steps,)
            matrices: Inertia matrix of each network
                (num_networks, num_steps, dim, dim)
            plot_dir: Directory to save plots to
            plot_first_step_only: Whether to plot only the first step
        """
        # precondition checks
        num_networks, num_steps, dim, _ = matrices.shape
        assert num_networks == len(network_names), \
            f"num_networks: {num_networks}, len(network_names): {len(network_names)}"
        assert num_steps == len(times), \
            f"num_steps: {num_steps}, len(times): {len(times)}"

        # create directory if it doesn't exist
        save_dir = f"{plot_dir}/{self.matrix_name}_eigvals"
        os.makedirs(save_dir, exist_ok=True)

        # compute eigenvalues. Store real/imaginary parts separately
        eigvals_real = np.zeros((num_networks, num_steps, dim))
        eigvals_imag = np.zeros((num_networks, num_steps, dim))
        for step_idx in range(num_steps):
            for nw_idx in range(num_networks):
                eigvals = np.linalg.eigvals(matrices[nw_idx, step_idx])
                eigvals_real[nw_idx, step_idx] = np.real(eigvals)
                eigvals_imag[nw_idx, step_idx] = np.imag(eigvals)

        # set up figure dimensions
        matplotlib.use('Agg')
        matplotlib.rcParams['figure.dpi'] = 100
        dpi = matplotlib.rcParams['figure.dpi']
        num_rows = 2
        num_cols = num_networks
        fig_w, fig_h = (700*num_cols+100, 700*num_rows)
        figsize = fig_w / float(dpi), fig_h / float(dpi)
        width_ratios = [1.0] * num_cols
        
        # set figure axes limits
        # NOTE: we want to center both the Re/Im axis at 0. And we assume that
        # most predicted M's should have eigenvalues on the real, positive axis
        x_width = np.amax(
            [np.abs(np.amax(eigvals_real)), np.abs(np.amin(eigvals_real))])
        x_max = 1.05 * x_width
        x_min = -1.05 * x_width
        if np.amax(eigvals_imag) - np.amin(eigvals_imag) < 1e-3:
            # if max imaginary part is close to min imaginary part, set y-axis
            # range same as x-axis range
            y_max = x_max
            y_min = x_min
        else:
            if np.amax(eigvals_imag) > 0.0:
                y_max = 1.05 * np.amax(eigvals_imag)
                y_min = -1.05 * np.amax(eigvals_imag)
            else:
                # if all imaginary parts are negative, set y-axis range same as
                # x-axis range
                y_max = x_max
                y_min = x_min

        for step_idx in range(num_steps):
            fig, axs = plt.subplots(
                nrows=num_rows, ncols=num_cols,
                figsize=figsize,
                squeeze=False,
                width_ratios=width_ratios)
            
            for nw_idx in range(num_networks):                
                for row_idx in range(num_rows):
                    # plot eigenvalues
                    axs[row_idx][nw_idx].scatter(
                        eigvals_real[nw_idx, step_idx],
                        eigvals_imag[nw_idx, step_idx],
                        marker="x", color="navy")
                    
                    # set axis limits (unzoomed plot)
                    if row_idx == 0:
                        axs[row_idx][nw_idx].set_xlim(x_min, x_max)
                        axs[row_idx][nw_idx].set_ylim(y_min, y_max)
                        
                    # set axis labels
                    axs[row_idx][nw_idx].set_xlabel("Re.")
                    axs[row_idx][nw_idx].set_ylabel("Im.")
                    
                    # annotate max/min real parts (unzoomed plot)
                    if row_idx == 0:
                        re_min = np.amin(eigvals_real[nw_idx, step_idx])
                        re_max = np.amax(eigvals_real[nw_idx, step_idx])
                        axs[row_idx][nw_idx].text(
                            0.35 * x_max,
                            0.35 * y_max,
                            f'(Re) min: {re_min:.2f}\n(Re) max: {re_max:.2f}',
                            color='black', 
                            bbox=dict(facecolor='white', edgecolor='black'))
                    
                    # set axis grid
                    # colors from https://jonathansoma.com/lede/data-studio/matplotlib/
                    # adding-grid-lines-to-a-matplotlib-chart/
                    axs[row_idx][nw_idx].grid(
                        which='major',
                        axis='x',
                        linestyle='-', linewidth='0.5', color='red')
                    axs[row_idx][nw_idx].grid(
                        which='minor',
                        axis='x',
                        linestyle=':', linewidth='0.5', color='black')
                    axs[row_idx][nw_idx].set_axisbelow(True)
                    # required for the minor grid to show up
                    axs[row_idx][nw_idx].minorticks_on()

                    # set axis title
                    if row_idx == 0:
                        axs[row_idx][nw_idx].set_title(network_names[nw_idx])
                    else:
                        axs[row_idx][nw_idx].set_title(
                            f"{network_names[nw_idx]} (zoomed)")
            
            # save the plot
            img_path = f"{save_dir}/{self.matrix_name}_eigvals+time={times[step_idx]:03d}ms.png"
            plt.savefig(img_path)
            plt.close(fig)

            # if plotting only the first step, break
            if plot_first_step_only:
                break

        if plot_first_step_only:
            print(f"Inertia eigenvalue plots of the first step saved to {plot_dir}")
        else:
            print(f"Inertia eigenvalue plots of {num_steps} steps saved to {plot_dir}")