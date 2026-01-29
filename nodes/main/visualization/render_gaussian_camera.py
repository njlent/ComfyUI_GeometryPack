# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Render Gaussian Splatting PLY files to images using gsplat.

Optimized for GPU rendering with gsplat library.
"""

import torch
import numpy as np
import os
import math
from plyfile import PlyData

class RenderGaussianCameraNode:
    """
    Render a Gaussian Splatting PLY file from a user-defined camera.
    
    Uses gsplat for efficient GPU rasterization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file"
                }),
                "width": ("INT", {
                    "default": 800, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Output image width"
                }),
                "height": ("INT", {
                    "default": 600, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Output image height"
                }),
                # Camera Transform
                "translate_x": ("FLOAT", {
                    "default": 0.0, "step": 0.1,
                    "tooltip": "Camera position X"
                }),
                "translate_y": ("FLOAT", {
                    "default": 0.0, "step": 0.1,
                    "tooltip": "Camera position Y"
                }),
                "translate_z": ("FLOAT", {
                    "default": 3.0, "step": 0.1,
                    "tooltip": "Camera position Z"
                }),
                # Camera Rotation
                "rotate_x": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Camera rotation X (Pitch) in degrees"
                }),
                "rotate_y": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Camera rotation Y (Yaw) in degrees"
                }),
                "rotate_z": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Camera rotation Z (Roll) in degrees"
                }),
                # Camera Intrinsics
                "focal_length": ("FLOAT", {
                    "default": 500.0, "min": 10.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Focal length in pixels"
                }),
                "background_color": (["black", "white", "transparent"], {
                    "default": "black",
                    "tooltip": "Background color for rendering"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "geompack/visualization"

    def render(self, ply_path, width, height,
               translate_x, translate_y, translate_z,
               rotate_x, rotate_y, rotate_z,
               focal_length, background_color="black"):
        
        # 1. Check for gsplat availability
        try:
            from gsplat import rasterization
        except ImportError:
            raise ImportError("The 'gsplat' library is required for this node. "
                              "Please install it using your ComfyUI environment's pip: "
                              "pip install gsplat")

        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        # 2. Load PLY data
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        
        # 3. Extract Gaussian parameters
        # Positions
        means = np.stack((vertex['x'], vertex['y'], vertex['z']), axis=-1)
        
        # Opacities
        opacities = vertex['opacity']
        # Apply sigmoid to raw opacities if they aren't already 0-1 (3DGS usually stores logit, but let's check standard)
        # Standard implementation usually stores sigmoid(opacity) or opacity directly?
        # Actually gsplat expects raw values generally, but 3DGS PLY standard is often `opacity` property which is passed through sigmoid.
        # Wait, standard 3DGS code applies sigmoid during loading. BUT `gsplat` might expect pre-activated or post-activated.
        # Checking gsplat docs: "opacities - The opacities of the Gaussians. [..., N]". Usually expects 0-1 values.
        # Standard PLY stores opacities passed through inverse sigmoid (logits).
        # Let's assume standard behavior: apply sigmoid.
        opacities = 1.0 / (1.0 + np.exp(-opacities))
        
        # Scales
        scale_names = [p.name for p in vertex.properties if p.name.startswith('scale_')]
        scale_names = sorted(scale_names) # scale_0, scale_1, scale_2
        scales = np.stack([vertex[n] for n in scale_names], axis=-1)
        # Standard PLY stores log(scale). Apply exp.
        scales = np.exp(scales)
        
        # Rotations (Quaternions)
        rot_names = [p.name for p in vertex.properties if p.name.startswith('rot_')]
        rot_names = sorted(rot_names) # rot_0, rot_1, rot_2, rot_3
        quats = np.stack([vertex[n] for n in rot_names], axis=-1)
        # Ensure normalization happens (gsplat docs say not required to be normalized, but good practice)
        # Normalize: q = q / norm(q)
        norms = np.linalg.norm(quats, axis=-1, keepdims=True)
        quats = quats / norms
        
        # Colors (Spherical Harmonics - DC component)
        # For simplicity, we'll just use the DC component (f_dc_0, f_dc_1, f_dc_2) which is the base color
        # This ignores view-dependent effects (SH degree 0)
        # Standard PLY has f_dc_0, f_dc_1, f_dc_2 corresponding to R, G, B coefficients
        # SH_C0 = 0.28209479177387814
        # RGB = 0.5 + SH * SH_C0  (approximate? No, standard formula is: rgb = 0.5 + sh * 0.28209)
        # Actually, let's look at how official renderers do it. 3DGS code:
        # features_dc = features[:,:,0:1] -> transpose -> flatten
        # basic color = SH_0 * f_dc + 0.5
        features_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
        SH_C0 = 0.28209479177387814
        colors = features_dc * SH_C0 + 0.5
        # Clip to 0-1
        colors = np.clip(colors, 0.0, 1.0)
        
        # 4. Prepare Tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        means = torch.tensor(means, dtype=torch.float32, device=device)
        quats = torch.tensor(quats, dtype=torch.float32, device=device)
        scales = torch.tensor(scales, dtype=torch.float32, device=device)
        opacities = torch.tensor(opacities, dtype=torch.float32, device=device)
        colors = torch.tensor(colors, dtype=torch.float32, device=device)
        
        # 5. Construct Camera Matrices
        # Camera Intrinsics (K)
        # [fx  0 cx]
        # [ 0 fy cy]
        # [ 0  0  1]
        cx = width / 2.0
        cy = height / 2.0
        fx = focal_length
        fy = focal_length
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        # Camera Extrinsics (World-to-Camera)
        # First construct Camera-to-World (C2W)
        # Rotation Matrix from Euler angles (check order: usually ZYX or XYZ)
        # Let's stick to standard pitch(x), yaw(y), roll(z)
        R = self._euler_to_rotation_matrix(rotate_x, rotate_y, rotate_z)
        T = np.array([translate_x, translate_y, translate_z])
        
        # World-to-Camera (W2C) = inverse(C2W)
        # R_w2c = R_c2w.T
        # T_w2c = -R_w2c @ T_c2w
        R_w2c = R.T
        T_w2c = -R_w2c @ T
        
        # Create 4x4 View Matrix
        viewmat = np.eye(4)
        viewmat[:3, :3] = R_w2c
        viewmat[:3, 3] = T_w2c
        
        viewmat = torch.tensor(viewmat, dtype=torch.float32, device=device)
        
        # Add batch dimension to standard inputs
        viewmats = viewmat.unsqueeze(0) # (1, 4, 4)
        Ks = K.unsqueeze(0) # (1, 3, 3)
        
        # 6. Render
        # gsplat.rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height)
        # Note: 'colors' here is used if sh_degree is NOT specified or None.
        
        # Background color
        bg_color = None
        if background_color == "white":
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
        elif background_color == "black":
            bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)
        # If transparent, we need to handle alpha, but gsplat usually returns RGB or RGBA?
        # Docs say: returns (tuples) ...
        # render_mode="RGB" -> returns (B, H, W, 3)
        # If we want transparency, maybe we need "RGB+A" or check output format.
        # gsplat returns [..., C, H, W, D].
        # Let's stick to RGB for now.
        
        image, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            backgrounds=bg_color if bg_color is not None else torch.zeros(3, device=device)
        )
        
        # image shape: (1, H, W, 3)
        # alpha shape: (1, H, W, 1)
        
        # 7. Post-process
        # If transparent background requested, combine with alpha
        if background_color == "transparent":
            # Concat alpha channel
            image = torch.cat([image, alpha], dim=-1) # (1, H, W, 4)
            
        # Ensure float32, 0-1 range (gsplat output should already be float)
        image = image.cpu()
        
        return (image,)

    def _euler_to_rotation_matrix(self, rx, ry, rz):
        """
        Create 3x3 rotation matrix from Euler angles (degrees).
        Order: Z (roll) -> Y (yaw) -> X (pitch)
        Matches standard camera conventions.
        """
        rx = math.radians(rx)
        ry = math.radians(ry)
        rz = math.radians(rz)
        
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        
        # Rx
        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])
        
        # Ry
        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])
        
        # Rz
        Rz = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])
        
        # R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx


NODE_CLASS_MAPPINGS = {
    "GeomPackRenderGaussianCamera": RenderGaussianCameraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRenderGaussianCamera": "Render Gaussian (Camera)",
}
