
import subprocess
import os
import sys

def get_blender_path():
    possible_paths = [
        "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("Blender executable not found")

def run_visualizer():
    blender_path = get_blender_path()
    blend_file = os.path.join(os.path.dirname(sys.executable), "visualizer.blend")
    
    subprocess.Popen([
        blender_path,
        "-b",  
        blend_file,
        "--python-exit-code", "1"
    ])

if __name__ == "__main__":
    run_visualizer()
