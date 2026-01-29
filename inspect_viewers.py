try:
    from comfy_3d_viewers import get_js_dir, get_nodes_dir
    print(f"JS_DIR: {get_js_dir()}")
    print(f"NODES_DIR: {get_nodes_dir()}")
except ImportError:
    print("ERROR: comfy_3d_viewers not installed")
except Exception as e:
    print(f"ERROR: {e}")
