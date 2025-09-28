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
    def __init__(self, graph: GraphModel, data_matrix: np.ndarray):
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
        self.vx = GraphSignal(self.graph, self._get_axis(self.velocities, 'x'))
        self.vy = GraphSignal(self.graph, self._get_axis(self.velocities, 'y'))
        self.vz = GraphSignal(self.graph, self._get_axis(self.velocities, 'z'))

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


