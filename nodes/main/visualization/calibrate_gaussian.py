# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Calibrate Gaussian Camera Node.

Allows interactive calibration of camera parameters by verifying the
3D viewport camera state.
"""

from .preview_gaussian import PreviewGaussianNode

class CalibrateGaussianCameraNode(PreviewGaussianNode):
    """
    Interactive calibration tool for Gaussian Splatting cameras.
    
    Displays the Gaussian Splat and exposes real-time camera parameters
    (Translation, Rotation, Focal Length) that can be copied to the
    RenderGaussianCamera node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file"
                }),
            },
            "optional": {
                "extrinsics": ("EXTRINSICS", {
                    "tooltip": "Initial 4x4 camera extrinsics matrix"
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Initial 3x3 camera intrinsics matrix"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "calibrate_gaussian"
    CATEGORY = "geompack/visualization"

    def calibrate_gaussian(self, ply_path, extrinsics=None, intrinsics=None):
        # reuse the preview logic
        result = super().preview_gaussian(ply_path, extrinsics, intrinsics)
        
        # Mark this as a calibration node for the frontend
        if "ui" in result:
            result["ui"]["mode"] = ["calibration"]
            
        return result


NODE_CLASS_MAPPINGS = {
    "GeomPackCalibrateGaussianCamera": CalibrateGaussianCameraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCalibrateGaussianCamera": "Calibrate Gaussian (Camera)",
}
