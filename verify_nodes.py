import sys
import os
import traceback

# Add nodes directory to path
sys.path.append(os.path.abspath("e:/ai/ComfyUI_DEV/Comfyui_windows_portable_dev05/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-GeometryPack/nodes"))

print("--- Verifying Imports ---")

try:
    print("Importing render_gaussian_camera...")
    import main.visualization.render_gaussian_camera as rgc
    print("SUCCESS: RenderGaussianCameraNode mappings:", rgc.NODE_CLASS_MAPPINGS)
    
    print("\nImporting calibrate_gaussian...")
    import main.visualization.calibrate_gaussian as cgc
    print("SUCCESS: CalibrateGaussianCameraNode mappings:", cgc.NODE_CLASS_MAPPINGS)
    
    print("\nImporting package __init__...")
    import main.visualization as pkg
    if "GeomPackCalibrateGaussianCamera" in pkg.NODE_CLASS_MAPPINGS:
        print("SUCCESS: Calibrate node found in package mappings!")
    else:
        print("FAILURE: Calibrate node NOT found in package mappings.")
        print("Keys:", pkg.NODE_CLASS_MAPPINGS.keys())

except Exception:
    traceback.print_exc()
