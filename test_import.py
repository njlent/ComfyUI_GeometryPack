import sys
import os

# Add nodes directory to path to simulate ComfyUI environment
sys.path.append(os.path.abspath("e:/ai/ComfyUI_DEV/Comfyui_windows_portable_dev05/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-GeometryPack/nodes"))

try:
    from main.visualization.render_gaussian_camera import RenderGaussianCameraNode
    print("SUCCESS: Imported RenderGaussianCameraNode")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
except Exception as e:
    print(f"ERROR: {e}")
