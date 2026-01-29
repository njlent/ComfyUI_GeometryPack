# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Combine Gaussians Node - Combine two 3D Gaussian Splatting PLY files into one.

Supports independent transform controls for each input Gaussian, allowing
precise positioning and rotation of splats in the combined scene.
"""

import os
import numpy as np
from plyfile import PlyData, PlyElement

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None


def quaternion_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def euler_to_quaternion(rx, ry, rz):
    """Convert Euler angles (degrees) to quaternion (w, x, y, z)."""
    rx_rad = np.radians(rx) / 2
    ry_rad = np.radians(ry) / 2
    rz_rad = np.radians(rz) / 2

    cx, sx = np.cos(rx_rad), np.sin(rx_rad)
    cy, sy = np.cos(ry_rad), np.sin(ry_rad)
    cz, sz = np.cos(rz_rad), np.sin(rz_rad)

    # ZYX order (matches transform.py)
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz

    return np.array([w, x, y, z])


def rotation_matrix_from_euler(rx, ry, rz):
    """Create 3x3 rotation matrix from Euler angles (degrees)."""
    rx_rad = np.radians(rx)
    ry_rad = np.radians(ry)
    rz_rad = np.radians(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_rad), -np.sin(rx_rad)],
        [0, np.sin(rx_rad), np.cos(rx_rad)]
    ])

    Ry = np.array([
        [np.cos(ry_rad), 0, np.sin(ry_rad)],
        [0, 1, 0],
        [-np.sin(ry_rad), 0, np.cos(ry_rad)]
    ])

    Rz = np.array([
        [np.cos(rz_rad), -np.sin(rz_rad), 0],
        [np.sin(rz_rad), np.cos(rz_rad), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


class CombineGaussiansNode:
    """
    Combine two 3D Gaussian Splatting PLY files into one.

    Each input Gaussian can be independently transformed (translate, rotate, scale)
    to position it correctly in the combined scene.

    Outputs the path to the combined PLY file along with camera parameters
    for use with the Preview Gaussian node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gaussian_1_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to first Gaussian Splatting PLY file"
                }),
                "gaussian_2_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to second Gaussian Splatting PLY file"
                }),
                "output_filename": ("STRING", {
                    "default": "combined_gaussian",
                    "tooltip": "Output filename (without extension)"
                }),
            },
            "optional": {
                # Gaussian 1 Transform
                "g1_translate_x": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Gaussian 1 translation X"
                }),
                "g1_translate_y": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Gaussian 1 translation Y"
                }),
                "g1_translate_z": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Gaussian 1 translation Z"
                }),
                "g1_rotate_x": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Gaussian 1 rotation X (degrees)"
                }),
                "g1_rotate_y": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Gaussian 1 rotation Y (degrees)"
                }),
                "g1_rotate_z": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Gaussian 1 rotation Z (degrees)"
                }),
                "g1_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 100.0, "step": 0.01,
                    "tooltip": "Gaussian 1 uniform scale"
                }),
                # Gaussian 2 Transform (default: rotated 180° and offset back)
                "g2_translate_x": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Gaussian 2 translation X"
                }),
                "g2_translate_y": ("FLOAT", {
                    "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Gaussian 2 translation Y"
                }),
                "g2_translate_z": ("FLOAT", {
                    "default": -0.5, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Gaussian 2 translation Z"
                }),
                "g2_rotate_x": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Gaussian 2 rotation X (degrees)"
                }),
                "g2_rotate_y": ("FLOAT", {
                    "default": 180.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Gaussian 2 rotation Y (degrees)"
                }),
                "g2_rotate_z": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Gaussian 2 rotation Z (degrees)"
                }),
                "g2_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 100.0, "step": 0.01,
                    "tooltip": "Gaussian 2 uniform scale"
                }),
                # Camera parameters (passthrough)
                "extrinsics": ("EXTRINSICS", {
                    "tooltip": "Camera extrinsics for preview (passthrough)"
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics for preview (passthrough)"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "EXTRINSICS", "INTRINSICS", "STRING")
    RETURN_NAMES = ("ply_path", "extrinsics", "intrinsics", "info")
    FUNCTION = "combine_gaussians"
    CATEGORY = "geompack/combine"
    OUTPUT_NODE = True

    def _load_gaussian_ply(self, path):
        """Load a Gaussian Splatting PLY file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Gaussian PLY file not found: {path}")

        plydata = PlyData.read(path)
        vertex = plydata['vertex']

        return plydata, vertex

    def _transform_gaussian(self, vertex, tx, ty, tz, rx, ry, rz, scale):
        """Apply transform to Gaussian splat data."""
        n_points = len(vertex.data)

        # Extract positions
        x = vertex['x'].copy()
        y = vertex['y'].copy()
        z = vertex['z'].copy()

        # Apply scale to positions
        x *= scale
        y *= scale
        z *= scale

        # Apply rotation to positions
        if rx != 0 or ry != 0 or rz != 0:
            rot_matrix = rotation_matrix_from_euler(rx, ry, rz)
            positions = np.column_stack([x, y, z])
            rotated = positions @ rot_matrix.T
            x, y, z = rotated[:, 0], rotated[:, 1], rotated[:, 2]

        # Apply translation
        x += tx
        y += ty
        z += tz

        # Create new vertex data with transformed positions
        # First, get all property names and types
        props = vertex.data.dtype.names
        new_data = np.zeros(n_points, dtype=vertex.data.dtype)

        # Copy all data
        for prop in props:
            new_data[prop] = vertex.data[prop]

        # Update positions
        new_data['x'] = x
        new_data['y'] = y
        new_data['z'] = z

        # Transform scales if present (apply uniform scale)
        if scale != 1.0:
            for scale_prop in ['scale_0', 'scale_1', 'scale_2']:
                if scale_prop in props:
                    # Scales are in log space for 3DGS
                    new_data[scale_prop] = vertex.data[scale_prop] + np.log(scale)

        # Transform rotations (quaternions) if rotation applied
        if rx != 0 or ry != 0 or rz != 0:
            if all(f'rot_{i}' in props for i in range(4)):
                euler_quat = euler_to_quaternion(rx, ry, rz)
                for i in range(n_points):
                    orig_quat = np.array([
                        vertex.data['rot_0'][i],
                        vertex.data['rot_1'][i],
                        vertex.data['rot_2'][i],
                        vertex.data['rot_3'][i]
                    ])
                    new_quat = quaternion_multiply(euler_quat, orig_quat)
                    # Normalize
                    new_quat = new_quat / np.linalg.norm(new_quat)
                    new_data['rot_0'][i] = new_quat[0]
                    new_data['rot_1'][i] = new_quat[1]
                    new_data['rot_2'][i] = new_quat[2]
                    new_data['rot_3'][i] = new_quat[3]

        return new_data

    def combine_gaussians(
        self, gaussian_1_path, gaussian_2_path, output_filename,
        g1_translate_x=0.0, g1_translate_y=0.0, g1_translate_z=0.0,
        g1_rotate_x=0.0, g1_rotate_y=0.0, g1_rotate_z=0.0, g1_scale=1.0,
        g2_translate_x=0.0, g2_translate_y=0.0, g2_translate_z=-0.5,
        g2_rotate_x=0.0, g2_rotate_y=180.0, g2_rotate_z=0.0, g2_scale=1.0,
        extrinsics=None, intrinsics=None
    ):
        """
        Combine two Gaussian Splatting PLY files.

        Args:
            gaussian_1_path: Path to first Gaussian PLY
            gaussian_2_path: Path to second Gaussian PLY
            output_filename: Output filename (without extension)
            g1_*/g2_*: Transform parameters for each Gaussian
            extrinsics/intrinsics: Camera parameters (passthrough)

        Returns:
            tuple: (ply_path, extrinsics, intrinsics, info)
        """
        print(f"[CombineGaussians] Loading Gaussian 1: {gaussian_1_path}")
        ply1, vertex1 = self._load_gaussian_ply(gaussian_1_path)

        print(f"[CombineGaussians] Loading Gaussian 2: {gaussian_2_path}")
        ply2, vertex2 = self._load_gaussian_ply(gaussian_2_path)

        n_points_1 = len(vertex1.data)
        n_points_2 = len(vertex2.data)
        print(f"[CombineGaussians] Gaussian 1: {n_points_1:,} points")
        print(f"[CombineGaussians] Gaussian 2: {n_points_2:,} points")

        # Transform each Gaussian
        print(f"[CombineGaussians] Applying transform to Gaussian 1: "
              f"translate=({g1_translate_x}, {g1_translate_y}, {g1_translate_z}), "
              f"rotate=({g1_rotate_x}, {g1_rotate_y}, {g1_rotate_z}), scale={g1_scale}")
        data1 = self._transform_gaussian(
            vertex1, g1_translate_x, g1_translate_y, g1_translate_z,
            g1_rotate_x, g1_rotate_y, g1_rotate_z, g1_scale
        )

        print(f"[CombineGaussians] Applying transform to Gaussian 2: "
              f"translate=({g2_translate_x}, {g2_translate_y}, {g2_translate_z}), "
              f"rotate=({g2_rotate_x}, {g2_rotate_y}, {g2_rotate_z}), scale={g2_scale}")
        data2 = self._transform_gaussian(
            vertex2, g2_translate_x, g2_translate_y, g2_translate_z,
            g2_rotate_x, g2_rotate_y, g2_rotate_z, g2_scale
        )

        # Combine data
        combined_data = np.concatenate([data1, data2])
        print(f"[CombineGaussians] Combined: {len(combined_data):,} total points")

        # Create new PLY element
        combined_vertex = PlyElement.describe(combined_data, 'vertex')

        # Determine output path
        if not output_filename.lower().endswith('.ply'):
            output_filename = output_filename + '.ply'

        if COMFYUI_OUTPUT_FOLDER and not os.path.isabs(output_filename):
            output_path = os.path.join(COMFYUI_OUTPUT_FOLDER, output_filename)
        else:
            output_path = output_filename

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Save combined PLY
        combined_ply = PlyData([combined_vertex], text=False)
        combined_ply.write(output_path)
        print(f"[CombineGaussians] Saved combined Gaussian to: {output_path}")

        # Build info string
        info = f"""Combine Gaussians Results:

Input Files:
  Gaussian 1: {os.path.basename(gaussian_1_path)} ({n_points_1:,} points)
  Gaussian 2: {os.path.basename(gaussian_2_path)} ({n_points_2:,} points)

Transforms Applied:
  Gaussian 1:
    Translate: ({g1_translate_x:.3f}, {g1_translate_y:.3f}, {g1_translate_z:.3f})
    Rotate: ({g1_rotate_x:.1f}°, {g1_rotate_y:.1f}°, {g1_rotate_z:.1f}°)
    Scale: {g1_scale:.3f}
  Gaussian 2:
    Translate: ({g2_translate_x:.3f}, {g2_translate_y:.3f}, {g2_translate_z:.3f})
    Rotate: ({g2_rotate_x:.1f}°, {g2_rotate_y:.1f}°, {g2_rotate_z:.1f}°)
    Scale: {g2_scale:.3f}

Combined Result:
  Total Points: {len(combined_data):,}
  Output: {output_path}
"""

        return (output_path, extrinsics, intrinsics, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackCombineGaussians": CombineGaussiansNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCombineGaussians": "Combine Gaussians",
}
