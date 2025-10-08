# src/gait_gft.py
"""
gait_gft: utilities for Graph Signal Processing under Gait Analysis context
Clases: GraphModel, SkeletonPreprocessor, GraphSignal, GaitTrial
"""

import os
import zipfile
from io import BytesIO
from typing import Tuple

import pandas as pd
import numpy as np
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.lines import Line2D



from scipy.interpolate import interp1d

class GraphModel:
    def __init__(self, edges: list[tuple[str, str]],
                 weights: dict[tuple[str, str], float] = None,
                 node_list: list[str] = None):

        """
        Parameters
        ----------
        edges : list of (node1, node2)
            Edge list defining graph connections.
        weights : optional dict of (node1, node2) → float
            Edge weights.
        node_list : optional list of str
            Explicit node ordering for consistent matrix representations.
        """

        self.graph = nx.Graph()
        self.node_list = node_list

        if self.node_list is not None:
            self.graph.add_nodes_from(self.node_list)
        else:
            self.node_list = list(sorted(self.graph.nodes))

        self.graph.add_edges_from(edges)

        if weights:
            nx.set_edge_attributes(self.graph, values=weights, name='weight')


        self.n = len(self.node_list)

        # self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        # self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}

        # Build matrices
        self.adj_matrix = nx.to_numpy_array(self.graph, nodelist=self.node_list, weight='weight')
        self.deg_matrix = np.diag(self.adj_matrix.sum(axis=1))
        self.laplacian = self.deg_matrix - self.adj_matrix
        self.norm_laplacian = self.compute_normalized_laplacian()

        # Spectral decomposition
        self.Lambda , self.U = self.compute_normal_modes()

    def compute_normalized_laplacian(self):
        d_inv_sqrt = np.diag(1 / np.sqrt(np.diag(self.deg_matrix)))
        return d_inv_sqrt @ self.laplacian @ d_inv_sqrt

    def compute_normal_modes(self):
        lambdas, U = np.linalg.eigh(self.norm_laplacian)
        return lambdas, U


class SkeletonPreprocessor:
    def __init__(self,
                 zip_path: str,
                 node_list: list[str],
                 time_thresholds: dict,
                 resample_freq: str = '30ms'):
        """
        Parameters
        ----------
        zip_path : str
            Path to the .zip file containing skeleton CSVs.
        node_list : list[str]
            Names of skeleton joints.
        time_thresholds : dict
            Dict containing time crop intervals per subject/pathology/trial.
        resample_freq : str
            Frequency for time resampling (e.g., '20ms' for 50Hz).
        """
        self.zip_path = zip_path
        self.node_list = node_list
        self.cols = pd.MultiIndex.from_tuples(
            [(joint, coord) for joint in self.node_list for coord in ['x', 'y', 'z']]
        )
        self.time_thresholds = time_thresholds
        self.resample_freq = resample_freq

    def _extract_meta(self, path: str) -> Tuple[str, str, str]:
        parts = path.split('/')
        subject, pathology, trial = parts[:3]
        return subject, pathology, trial

    def _get_time_bounds(self, subject: str, pathology: str, trial: str) -> Tuple[float, float]:
        thresholds = self.time_thresholds[subject][pathology]
        return thresholds.get(trial, thresholds['rest'])

    def load_and_process(self, internal_path: str) -> pd.DataFrame:
        """
        Load, parse, trim, deduplicate, and resample skeleton data from .zip.

        Parameters
        ----------
        internal_path : str
            Relative path inside the zip (e.g., 'subject10/normal/trial3/skeleton.csv').

        Returns
        -------
        pd.DataFrame
            Cleaned, trimmed, resampled DataFrame with MultiIndex columns and DatetimeIndex.
        """
        subject, pathology, trial = self._extract_meta(internal_path)
        lower, upper = self._get_time_bounds(subject, pathology, trial)

        # Read raw CSV from zip
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open(internal_path) as f:
                df = pd.read_csv(BytesIO(f.read()), header=None, index_col=0).iloc[:, :96]

        # Format index and column names
        df.index = pd.to_datetime(df.index.map(
            lambda x: x.replace('_', ' ').replace('-', '')
        ))
        df.columns = self.cols

        # Trim based on relative time
        n_rows = df.shape[0]
        pct_time = np.arange(n_rows) / n_rows
        mask = (pct_time >= lower) & (pct_time <= upper)
        df_trimmed = df[mask]

        # Average duplicates (same timestamp), then resample
        df_clean = df_trimmed.groupby(df_trimmed.index).mean()
        df_resampled = self._resample(df_clean)

        return df_resampled


    def _resample(self, df: pd.DataFrame, method: str = 'cubic') -> pd.DataFrame:
        """
        Resample a multi-index column DataFrame (joint, axis) over a regular time grid using smooth interpolation.

        Parameters
        ----------
        df : pd.DataFrame
            Original DataFrame with a DatetimeIndex and MultiIndex columns (joint, axis).
        target_freq : str
            Frequency string for the resampled index (e.g., '20ms' for 50Hz).
        method : str
            Interpolation method: 'linear', 'quadratic', 'cubic', 'slinear', 'akima', etc.

        Returns
        -------
        pd.DataFrame
            Resampled DataFrame on a regular time grid.
        """
        # 1. Sort and create regular time index
        df = df.sort_index()
        start, end = df.index[0], df.index[-1]
        new_index = pd.date_range(start, end, freq=self.resample_freq)

        # 2. Convert time index to numeric seconds
        original_times = df.index.view(np.int64) / 1e9
        target_times = new_index.view(np.int64) / 1e9

        # 3. Interpolate each column
        data = {}
        for col in df.columns:
            y = df[col].values
            f = interp1d(original_times, y, kind=method, fill_value='extrapolate')
            data[col] = f(target_times)

        # 4. Return resampled DataFrame
        resampled_df = pd.DataFrame(data, index=new_index)
        resampled_df.columns = df.columns  # preserve multiindex
        return resampled_df



class GraphSignal:
    def __init__(self, graph: GraphModel, data_matrix: np.ndarray, label:str = None):
        self.label = label
        self.graph = graph
        self.X = data_matrix  # shape: (n_nodes, n_timesteps)
        self.lambdas, self.U = self.graph.compute_normal_modes()
        self.X_hat = self.compute_gft()
        self.energy = self.compute_energy()

    def compute_gft(self):
        return self.U.T @ self.X

    def compute_energy(self):
        X = self.X
        L = self.graph.norm_laplacian
        E_t = pd.Series(np.einsum('ij,ji->i', X.T.values, L @ X.values), X.columns)
        return E_t

    def apply_filter(self, mode_indices):
        # Truncate to selected Fourier modes
        pass


class GaitTrial:
    def __init__(self,
                 position_df: pd.DataFrame,
                 graph: GraphModel,
                 label: str = None,
                 reference_joint: str = "SPINE_NAVAL"):
        self.label = label
        self.graph = graph
        self.reference_joint = reference_joint

        # Store positions (joint, axis)
        self.positions = position_df  # shape: (T, MultiIndex['joint', 'axis'])

        # Reference joint position over time (T × ['x', 'y', 'z'])
        self.reference_position = self._extract_reference_positions()

        # Centered position matrix (T × MultiIndex)
        self.centered_positions = self._center_positions()

        # Compute and store velocities
        self.velocities = self._compute_velocities(self.centered_positions)

        # Create GraphSignal objects per axis (3 directions)
        self.vx = GraphSignal(self.graph, self._get_axis(self.velocities, 'x'), label='x')
        self.vy = GraphSignal(self.graph, self._get_axis(self.velocities, 'y'), label='y')
        self.vz = GraphSignal(self.graph, self._get_axis(self.velocities, 'z'), label='z')

        self.gft_dict = dict(x=self.vx, y=self.vy, z=self.vz)

    def _extract_reference_positions(self) -> pd.DataFrame:
        """Extract position of reference joint over time (T × ['x', 'y', 'z'])"""
        return self.positions.loc[:, self.reference_joint]

    def _center_positions(self) -> pd.DataFrame:
        """Subtract reference joint from all joints."""
        centered = self.positions.copy()
        for axis in ['x', 'y', 'z']:
            centered_axis = centered.xs(axis, level=1, axis=1)
            centered_axis = centered_axis.sub(self.reference_position[axis], axis=0)
            centered_axis.columns = centered.loc[:, (slice(None), axis)].columns
            centered.loc[:, (slice(None), axis)] = centered_axis
        return centered

    def _compute_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute numerical derivative over time for all joints and axes."""
        dt = (df.index[1] - df.index[0]).total_seconds()  # assume constant dt
        return df.diff().dropna() / dt

    def _get_axis(self, df: pd.DataFrame, axis: str) -> pd.DataFrame:
        """Extract signal matrix for a single axis. Shape: (n_nodes, T)"""
        axis_df = df.xs(axis, level=1, axis=1)  # shape: (T, n_joints)
        return axis_df.T.loc[self.graph.node_list,:]  # shape: (n_joints, T)

    def get_position_matrix(self, axis: str) -> np.ndarray:
        """Return position matrix (n_nodes, T) for a specific axis."""
        return self._get_axis(self.positions, axis).values

    def get_velocity_matrix(self, axis: str) -> np.ndarray:
        """Return velocity matrix (n_nodes, T) for a specific axis."""
        return self._get_axis(self.velocities, axis).values



class Visualizer:
    def __init__(self):
        pass

    def plot_graph_basis(self, graph: GraphModel, figsize=(6.4, 8), cmap='seismic', save_path=None):
        """
        Plot the Laplacian eigenvectors (U matrix) as a heatmap and the eigenvalues as a line plot above.

        Parameters
        ----------
        graph : GraphModel
            GraphModel object with .U, .Lambda, .nodes attributes.
        figsize : tuple
            Size of the figure.
        cmap : str
            Colormap for the heatmap.
        save_path : str or None
            If provided, save the figure to this path.
        """
        U = graph.U
        lambdas = graph.Lambda
        nodes = graph.node_list
        n = graph.n

        fig, (ax1, ax2) = plt.subplots(
            nrows=2, figsize=figsize, sharex=True,
            gridspec_kw={'height_ratios': [1, 3]}
        )

        # --- Upper: Eigenvalues
        width = 0.1
        ax1.scatter(
            np.arange(1, n+1) - 0.5, lambdas,
            color='#9B0014', edgecolor='black', s=50, marker='o', alpha=0.8
        )
        ax1.set_ylabel("Eigenvalue", fontsize=12)
        ax1.set_title(r"$L_N$ Eigenvalues", fontsize=14, pad=10)
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.set_xlim(-1, len(lambdas) + 1)
        ax1.set_ylim(-width, max(2, lambdas.max()) + width)

        # --- Lower: Heatmap of U
        ax2.set_aspect('auto')
        sns.heatmap(U, cmap=cmap, center=0, cbar=False,
                    xticklabels=False, yticklabels=False, square=True, ax=ax2)

        ax2.set_xlabel("Eigenpair Index", fontsize=12)
        ax2.set_ylabel("Graph Node Index", fontsize=12)
        ax2.set_title(r"$L_N$ Eigenvectors", fontsize=14, pad=10)

        # Gridlines (vertical)
        for x in range(1, n):
            ax2.axvline(x, color='black', linewidth=0.7)

        ax2.set_xticks(np.arange(n) + 0.5)
        ax2.set_xticklabels(range(1, n + 1), fontsize=8, rotation=90)

        ax2.set_yticks(np.arange(n) + 0.5)
        ax2.set_yticklabels(nodes, fontsize=8)

        # --- External colorbar
        heatmap_box = ax2.get_position()
        cbar_height = 0.32
        cbar_bottom = heatmap_box.y0 + (heatmap_box.height - cbar_height) / 2
        cbar_ax = fig.add_axes([0.91, cbar_bottom, 0.02, cbar_height])

        norm = plt.Normalize(vmin=U.min(), vmax=U.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        f = 0.85
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[U.min()*f, 0, U.max()*f])
        cbar.ax.set_yticklabels(["-", "0", "+"], fontsize=12)

        # --- Layout
        plt.tight_layout(rect=[0, 0, 0.9, 1], h_pad=1.5)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def plot_graph_signal(self, signal,
                          ax,
                          graph: GraphModel,
                          pos: dict,
                          norm_edges=None,
                          font_size=3, edge_width=1,
                          edge_cmap=plt.cm.gray_r,
                          node_size=100,
                          nodes_cmap=plt.cm.seismic,
                          draw_labels=False,
                          grayscale_edges=True,
                          signal_min_max=None,
                          constant_color=None,
                          draw_edges=True,
                          rotation=0,
                          font_color='black',
                          ):
        """
        Plot a graph-signal on its graph representation.

        Parameters
        ----------
        signal : np.ndarray
            1D array of signal values, length = number of nodes.
        ax : matplotlib axis
            Axis on which to draw.
        graph : GraphModel
            Graph structure and node order.
        pos : dict
            Node position layout (dict of node → (x, y)).
        labels : dict
            Node labels for display.
        edges_weights : np.ndarray
            Edge weights (optional).
        norm_edges : Normalize object
            Normalizer for edge colors.
        All other parameters: passed to `nx.draw_networkx_...`
        """
        if pos is None:
            raise ValueError("A position dictionary `pos` must be provided.")

        labels=dict(zip(graph.node_list,graph.node_list))
        edges_weights = np.array([
            graph.graph[u][v].get('weight', 1.0)
            for u, v in graph.graph.edges()
        ])
        if norm_edges is None:
            norm_edges = plt.Normalize(edges_weights.min(), edges_weights.max())


        # Normalize node values
        if signal_min_max is None:
            max_abs_val = np.max(np.abs(signal))
            signal_min_max = (-max_abs_val, max_abs_val)
        else:
            val = max(abs(signal_min_max[0]), abs(signal_min_max[1]))
            signal_min_max = (-val, val)

        norm_signal = plt.Normalize(vmin=signal_min_max[0], vmax=signal_min_max[1])

        # Node colors
        if constant_color is not None:
            node_colors = constant_color
        else:
            node_colors = nodes_cmap(norm_signal(signal))

        # Draw nodes
        nx.draw_networkx_nodes(
            graph.graph, pos,
            node_color=node_colors,
            node_size=node_size,
            # cmap=nodes_cmap,
            ax=ax
        )

        # Draw edges
        if draw_edges:
            if grayscale_edges:
                nx.draw_networkx_edges(
                    graph.graph, pos,
                    edge_color=edges_weights,
                    edge_cmap=edge_cmap,
                    width=edge_width,
                    edge_vmin=min(0, edges_weights.min()),
                    edge_vmax=edges_weights.max(),
                    ax=ax
                )
            else:
                nx.draw_networkx_edges(graph.graph, pos, width=edge_width, ax=ax)

        # Draw labels
        if draw_labels:
            label_texts = nx.draw_networkx_labels(
                graph.graph, pos,
                font_size=font_size,
                font_weight='bold',
                labels=labels,
                ax=ax,
                font_color=font_color
            )
            if rotation != 0:
                for _, t in label_texts.items():
                    t.set_rotation(rotation)

        ax.axis("off")

        pos_X, pos_Y = zip(*pos.values())
        xlims = (min(pos_X)-10, max(pos_X)+10)
        ylims = (min(pos_Y)-10, max(pos_Y)+10)
    
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)


    def plot_normal_modes_grid(self, graph, pos,
                               n_rows=2, n_cols=4,
                               node_size=12,
                               figsize=(6.5, 4),
                               save_path_prefix='First normal modes',
                               **plot_kwargs):
        """
        Plot the first n_freqs = n_rows × n_cols graph Fourier modes in a grid layout.
    
        Parameters
        ----------
        graph : GraphModel
            Graph model with eigenvectors and structure.
        pos : dict
            Node layout positions.
        n_rows : int
            Number of subplot rows.
        n_cols : int
            Number of subplot columns.
        box_flag : bool
            Whether to draw boxes around selected modes.
        highlight_indices : list of int
            Mode indices to draw boxes around (zero-based).
        node_size : int
            Size of graph nodes.
        figsize : tuple
            Size of the full figure.
        save_path_prefix : str
            Prefix for saved filename.
        **plot_kwargs : additional kwargs passed to plot_graph_signal
        """
        U = graph.U
        n_freqs = n_rows * n_cols
        n_freqs = min(n_freqs, graph.n)  # Don't exceed available modes
    
        # Normalize colors across all shown modes
        u_min = U[:, :n_freqs].min()
        u_max = U[:, :n_freqs].max()
        max_val = max([abs(u_min), abs(u_max)])
        norm_u = plt.Normalize(-max_val, max_val)
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
    
        for i in range(n_freqs):
            ax = axes[i]
            signal = U[:, i]
    
            self.plot_graph_signal(
                signal,
                ax,
                graph,
                pos,
                node_size=node_size,
                signal_min_max=(-max_val, max_val),
                # **plot_kwargs
            )
    
    
            ax.set_title(f'k = {i + 1}', fontsize=10)
    
        # Hide unused axes
        for j in range(n_freqs, len(axes)):
            axes[j].axis("off")
    
        plt.tight_layout()
    
        # Save figure
        save_name = f"../results/{save_path_prefix}.png"
        fig.savefig(save_name, dpi=300)
        plt.show()

    
    def plot_energy_evolution(self, signal:GraphSignal , ax=None, label=None,  title=None, display_timestamps=False):
        """
        Plot the evolution of graph signal energy over time for each axis channel.

        Parameters
        ----------
        gft_dict : dict[str, GraphSignal]
            Dictionary with keys 'x', 'y', 'z' and GraphSignal objects as values.
        frame : int or None
            Optional timestep to highlight with a vertical dashed line.
        ax : matplotlib axis or None
            If None, a new figure and axis will be created.
        """

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(9,6))
        
        energy = signal.energy.copy()  # pd.Series indexed by time
        
        if not display_timestamps:
            energy.index = range(len(energy))
            
        sns.lineplot(x=energy.index, y=energy.values, ax=ax, label=label)

        ax.set_xlabel('Timestep', fontsize=15)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        ax.set_title("Graph-signal time evolution" if title is None else title, fontsize=16)

        
        ax.set_xlabel('Timestep', fontsize=10)
        
        if not display_timestamps:
            n_timesteps = len(energy)
            n_ticks = max(2, int(n_timesteps * 0.2))
            tick_positions = np.linspace(0, n_timesteps - 1, n_ticks, dtype=int)
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions, fontsize=8)


    def plot_heatmap(self, matrix, ax, v_max=None, include_gradient=True, display_timestamps=False):


        if not display_timestamps:
            matrix.columns = range(matrix.shape[1])

        n = matrix.shape[0]
        if v_max is None:
            v_max = np.abs(matrix.values).max()

        sns.heatmap(matrix, ax=ax, cmap='seismic', center=0, cbar=False,
                    vmin=-v_max, vmax=v_max,
                    yticklabels=False)
        
        ax.set_xlabel('Timestep', fontsize=10)

        ax.set_yticks(np.arange(n) + 0.5)

        if not display_timestamps:
            n_timesteps = matrix.shape[1]
            n_ticks = max(2, int(n_timesteps * 0.2))
            tick_positions = np.linspace(0, n_timesteps - 1, n_ticks, dtype=int)
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions, fontsize=8)
            
            


        if not display_timestamps:
            n_timesteps = matrix.shape[1]
            n_ticks = max(2, int(n_timesteps * 0.2))
            tick_positions = np.linspace(0, n_timesteps - 1, n_ticks, dtype=int)
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions, fontsize=8)

        if include_gradient:
            fig = ax.get_figure()
            gradient_ax = fig.add_axes([0.1, -0.05, 0.8, 0.025])  # Position for the colorbar
            gradient = np.linspace(-1, 1, 256).reshape(1, -1)  # Generate gradient data from -1 to 1
            gradient_ax.imshow(gradient, aspect='auto', cmap='seismic', extent=[-1, 1, 0, 1])
            gradient_ax.set_xticks([-1, 0, 1])
            gradient_ax.set_xticklabels(['-', '0', '+'], fontsize=4)
            gradient_ax.set_yticks([])
            gradient_ax.tick_params(axis='x', labelsize=20)

    
    def plot_graph_signal_matrix(self, signal: GraphSignal, ax=None, v_max=None, include_gradient=True, title=None, display_timestamps=False):
        matrix = signal.X.copy()
        y_labels = signal.graph.node_list

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        self.plot_heatmap(matrix, ax=ax, v_max=v_max, include_gradient=include_gradient, display_timestamps=display_timestamps)
        ax.set_title("Graph-signal through time" if title is None else title, fontsize=16)
        ax.set_yticklabels(y_labels, fontsize=8)

    
    def plot_gft(self, signal: GraphSignal, ax=None, v_max=None, include_gradient=True, title=None, display_timestamps=False):
        matrix = signal.X_hat.copy()
        y_labels = range(1, signal.graph.n+1)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        self.plot_heatmap(matrix, ax=ax, v_max=v_max, include_gradient=include_gradient, display_timestamps=display_timestamps)
        ax.set_title("GFT through time" if title is None else title, fontsize=16)
        ax.set_yticklabels(y_labels, fontsize=8)


    def plot_spectral_summary(self, signal: GraphSignal, axes=None, display_timestamps=False):

        if len(axes) != 2:
            print("Axes object should contain two elements!")
            return None
        elif axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            plt.subplots_adjust(wspace=0.3)
        else:
            fig = axes[0].get_figure()
        
        v_max = np.abs(signal.X_hat.values).max()
        
        label_size = 16
        tick_label_size = 8
        title_size = 19
        
        self.plot_energy_evolution(signal, ax=axes[0],  title=f"Energy $v_{{{signal.label}}}$", display_timestamps=display_timestamps)
        
        self.plot_gft(signal, ax=axes[1], include_gradient=False, title=f"GFT $v_{{{signal.label}}}$", display_timestamps=display_timestamps)
        
        axes[1].set_yticks(np.arange(32) + 0.5)
        odd_labels = [str(i) if i % 2 == 1 else "" for i in range(1, 33)]
        axes[1].set_yticklabels(odd_labels, fontsize=tick_label_size);
        
        
        # --- Add external colorbar ---
        # Create new axis for colorbar: [left, bottom, width, height]
        
        # Get the position of ax2 (the heatmap)
        heatmap_box = axes[1].get_position()
        
        # Calculate vertical center and adjust colorbar height/position accordingly
        cbar_height = 0.32  # You can tune this value
        cbar_bottom = heatmap_box.y0 + (heatmap_box.height - cbar_height) / 2
        
        # Create new axis for colorbar aligned to heatmap center
        cbar_ax = fig.add_axes([0.91, cbar_bottom, 0.02, cbar_height])
        
        #cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.32])
        norm = plt.Normalize(vmin=-v_max, vmax=v_max)
        sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
        sm.set_array([])
        f = 0.85
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[-v_max*f, 0, v_max*f])
        cbar.ax.set_yticklabels(["-", "0", "+"], fontsize=12);
        
        # # --- Finalize Layout ---
        # plt.tight_layout(rect=[0, 0, 0.9, 1], h_pad=1.5)  # leave space on the right for colorbar
        # # fig.savefig(f'gait_pathologies_compared.png', dpi=300, bbox_inches='tight')
        # plt.show()


class GaitVisualizer(Visualizer):
    """
    Specialized visualizer for gait analysis tasks.
    Inherits all general-purpose graph visualization methods.
    """

    def __init__(self):
        super().__init__()


    def set_axes_equal(self, ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.
    
        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        """
    
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
    
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
    
        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])
    
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



# Pending to add: velocity vectors (optional)
    def update_skeleton_frame(self, frame, gait: GaitTrial, ax, show_progress=True,
                              scaling_factor=0.5, display_velocity_vectors=True):
        
        if show_progress:
            print(f"{frame:>3}", end=", " + ("\n" if ((frame+1)%10)==0 else ""))
        ax.cla()  # Clear the previous frame to update the 3D plot

        total_frames = gait.positions.shape[0]
        frame = min([frame,total_frames-1])
        
        data    = gait.positions.iloc[frame].unstack().loc[gait.graph.node_list,:]
        frame_pos = data.apply(tuple,axis=1).to_dict()

        frame_v = min([frame, gait.velocities.shape[0]-1])
        v_frame = gait.velocities.iloc[frame_v].unstack().loc[gait.graph.node_list,:]
        
        
        
        # Nodes scatter plot
        x_coords = data['x'].values
        y_coords = data['y'].values
        z_coords = data['z'].values
        ax.scatter(x_coords, y_coords, z_coords, color='blue')


    
        if display_velocity_vectors:
            # Compute the magnitudes of the vectors
            v_magnitudes = np.linalg.norm(v_frame[['x', 'y', 'z']].values, axis=1)
        
            # Normalize the magnitudes for the colormap
            norm = Normalize(vmin=v_magnitudes.min(), vmax=v_magnitudes.max()*1.2)
            cmap = plt.colormaps['magma']  # Choose a gradient colormap (e.g., 'viridis', 'plasma')
        
            scaled_vectors = v_frame[['x', 'y', 'z']].values * scaling_factor
        
            # Plot the scaled vectors with gradient colors
            for i, (index, row) in enumerate(v_frame.iterrows()):
                start = data.loc[index, ['x', 'y', 'z']].values  # Starting point of the vector
                vector = scaled_vectors[i]  # Scaled vector components
                color = cmap(norm(v_magnitudes[i]))  # Color based on original magnitude
        
                ax.quiver(
                    start[0], start[1], start[2],  # Starting point
                    vector[0], vector[1], vector[2],  # Vector components
                    color=color, length=1.0, normalize=False  # Use scaled length
                )
    

        edge_xyz = np.array([
                (frame_pos[u], frame_pos[v])
                for u, v in gait.graph.graph.edges()
            ])
        
        # Edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")
        
        # Update title and axes labels
        ax.set_title(f"{gait.label.title()}", fontsize=20)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("y", fontsize=16)
        ax.set_zlabel("z", fontsize=16)
        
        self.set_axes_equal(ax)

    
    def walking_skeleton_animation(self, gait: GaitTrial,
                                   fps = 24, frames=None,
                                   display_velocity_vectors=True,
                                   show_progress=True,
                                   scaling_factor=0.5):

        print(f"Processing {gait.label.title()} gait trial animation:")
        fig, ax = plt.subplots(1,1, subplot_kw={'projection': '3d'}, figsize=(6, 4))
        ax.view_init(elev=30, azim=45, roll=0)

        if frames is None: frames = gait.positions.shape[0]
        interval = 1000 / fps  # interval in milliseconds
        
        ani = FuncAnimation(fig, self.update_skeleton_frame,
                            frames=frames,
                            interval=interval,
                            repeat=False,
                            fargs=(gait,ax,show_progress,
                                   scaling_factor,
                                   display_velocity_vectors))


        filepath = f"../results/walking_skeleton_{gait.label}_with{'' if display_velocity_vectors else 'out'}_vectors.gif"
        print(f"File saved at: {filepath}")
        ani.save(filepath, writer="ffmpeg")



    def plot_energy_evolution_all_components(self, gait: GaitTrial, ax=None, title=None, display_timestamps=False):

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(9,6))

        for label, signal in gait.gft_dict.items():
            self.plot_energy_evolution(signal, ax=ax, label=label, title=title, display_timestamps=display_timestamps)

    
    def plot_original_signal_vs_gft(self, gait: GaitTrial):
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True)
        plt.subplots_adjust(wspace=0.3)
        
        label_size = 16
        
        fig.suptitle(f"Graph-signals and their GFT:\n{gait.label.title()}", fontsize=22, y=0.96)
        
        for row, (channel, signal) in enumerate(gait.gft_dict.items()):
        
            title = f"$V_{channel}$"
            self.plot_graph_signal_matrix(signal, ax=axes[row,0], include_gradient=False, title=title, display_timestamps=False)
        
            title = f"$\hat{{V}}_{channel}$"
            self.plot_gft(signal, ax=axes[row,1], include_gradient=False, title=title)
        
            axes[row, 0].set_ylabel('Node', fontsize=label_size)
            axes[row, 1].set_ylabel('Normal mode', fontsize=label_size)
        
            if row == 2:
                axes[row, 0].set_xlabel('Timestep', fontsize=10)
                axes[row, 1].set_xlabel('Timestep', fontsize=10)
            else:
                axes[row, 0].set_xlabel('')
                axes[row, 1].set_xlabel('')
        
        
        
        # Define vertical span (0 = bottom of figure, 1 = top)
        v_min = 0.1  # adjust to leave some margin
        v_max = 0.9
        
        left_ax = axes[0, 0].get_position()
        right_ax = axes[0, 1].get_position()
        
        # Midpoint between right edge of left_ax and left edge of right_ax
        x_sep = (left_ax.x1 + right_ax.x0) * 0.48
        
        # Add vertical middle line
        fig.lines.append(plt.Line2D([x_sep, x_sep], [v_min, v_max], transform=fig.transFigure,
                                    color='black', linewidth=0.85))
        
        
        gradient_ax = fig.add_axes([0.1, 0.01, 0.8, 0.025])
        gradient = np.linspace(-1, 1, 256).reshape(1, -1)
        gradient_ax.imshow(gradient, aspect='auto', cmap='seismic', extent=[-1, 1, 0, 1])
        gradient_ax.set_xticks([-1, 0, 1])
        gradient_ax.set_xticklabels(['-', '0', '+'], fontsize=4)
        gradient_ax.set_yticks([])
        gradient_ax.tick_params(axis='x', labelsize=20)
        # fig.savefig(f'../results/original_signal_vs_gft.png', dpi=300, bbox_inches='tight')
        plt.show()


    def plot_compare_gait_types(self, gaits_dict: dict[str, GaitTrial], save_name: str = None):
        
        fig, axes = plt.subplots(4, len(gaits_dict), figsize=(35, 25))
        plt.subplots_adjust(wspace=0.3)
        
        label_size = 16        
        
        for col, (pathology, gait) in enumerate(gaits_dict.items()):
            
            pathology += ' gait' if pathology == 'normal' else ''
            title = f"{pathology.title()} energy"
            self.plot_energy_evolution_all_components(gait, ax=axes[0,col], title=title)
            for row, (channel, signal) in enumerate(gait.gft_dict.items(), start=1):
        
                title = f"$\hat{{V}}_{{{channel}}}$ - {pathology.title()}"
                self.plot_gft(signal, ax=axes[row,col], include_gradient=False, title=title)
        
                if col == 0:
                    axes[row, col].set_ylabel('Normal mode', fontsize=label_size)
                    axes[0,col].set_ylabel('Graph-signal Energy', fontsize=label_size)
        
                if row == 3:
                    axes[row, col].set_xlabel('Timestep', fontsize=10)
                    axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), rotation=90)
                else:
                    axes[row, col].set_xticklabels([], fontsize=label_size)
        
        # Define vertical span (0 = bottom of figure, 1 = top)
        v_min = 0.1  # adjust to leave some margin
        v_max = 0.9
        
        # Loop through the gaps between column 0–1 and column 1–2
        for col in range(len(gaits_dict)-1):
            left_ax = axes[0, col].get_position()
            right_ax = axes[0, col + 1].get_position()
        
            # Midpoint between right edge of left_ax and left edge of right_ax
            x_sep = (left_ax.x1 + right_ax.x0) / 2
        
            # Add vertical line
            fig.lines.append(plt.Line2D([x_sep, x_sep], [v_min, v_max], transform=fig.transFigure,
                                        color='black', linewidth=0.85))
        
        gradient_ax = fig.add_axes([0.1, 0.03, 0.8, 0.025])
        gradient = np.linspace(-1, 1, 256).reshape(1, -1)
        gradient_ax.imshow(gradient, aspect='auto', cmap='seismic', extent=[-1, 1, 0, 1])
        gradient_ax.set_xticks([-1, 0, 1])
        gradient_ax.set_xticklabels(['-', '0', '+'], fontsize=4)
        gradient_ax.set_yticks([])
        gradient_ax.tick_params(axis='x', labelsize=20)

        if save_name is not None:
            fig.savefig(f'../results/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()


    def animation_with_gft_summary(self, skeleton, gait, signal, display_velocity_vectors=True):
        
        fig = plt.figure(figsize=(15,5))
        
        n_modes = 2
        width_3d = 6
        width_2d = 2
        gs = fig.add_gridspec(n_modes, width_3d + 3*width_2d)
        
        ax = fig.add_subplot(gs[:, :width_3d], projection='3d')
        ax.view_init(elev=25, azim=45, roll=0)
        
        # axs  = [fig.add_subplot(gs[i, width_3d:]) for i in range(n_modes)]
        axs  = [fig.add_subplot(gs[0, width_3d:])]
        axs += [fig.add_subplot(gs[i, width_3d:], sharex=axs[0]) for i in range(1, n_modes)]
        
        self.plot_spectral_summary(signal, axs)
        
        
        # Create vertical lines manually
        line_energy = Line2D([0, 0], [0, 0], color='green', linestyle='--', linewidth=1.5)
        axs[0].add_line(line_energy)
        
        tip_marker = Line2D([0], [0],
                            marker='o',
                            markerfacecolor='white',
                            markeredgecolor='green',
                            markersize=10,
                            linestyle='None')
        axs[0].add_line(tip_marker)
        
        line_gft = axs[1].axvline(x=0, color='green', linestyle='--', linewidth=1.5)
        
        axs[0].tick_params(labelbottom=False)
        axs[0].set_xlabel("")
        
        # Function to update the scatter plot for each frame
        def update(frame, gait, ax, show_progress=True,
                   scaling_factor=0.5, display_velocity_vectors=True):
        
            frame = min([frame, len(signal.energy) - 1])
        
            self.update_skeleton_frame(frame, gait, ax, show_progress=show_progress,
                                       scaling_factor=scaling_factor, display_velocity_vectors=display_velocity_vectors)
        
            energy_val = signal.energy.iloc[frame]
            line_energy.set_data([frame, frame], [0, energy_val])
            tip_marker.set_data([frame], [energy_val])
            line_gft.set_xdata([frame])
        
        fps = 12  # frames per second
        interval = 1000 / fps  # interval in milliseconds
        
        ani = FuncAnimation(fig, update,
                            frames=len(signal.energy),
                            interval=interval, repeat=False,
                            fargs=(gait, ax))

        filepath = f"../results/skeleton_gft_animation_v{signal.label}.gif"
        print(f"File saved at: {filepath}")
        ani.save(filepath, writer="ffmpeg")

