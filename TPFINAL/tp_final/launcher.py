import PyInstaller.__main__
import os

def create_visualizer_exe():
    with open("run_visualizer.py", "w") as f:
        f.write("""
import subprocess
import os
import sys

def get_blender_path():
    # Add your Blender installation path here
    possible_paths = [
        "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe",
        "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
        # Add other possible paths
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("Blender executable not found")

def run_visualizer():
    blender_path = get_blender_path()
    blend_file = os.path.join(os.path.dirname(sys.executable), "visualizer.blend")
    
    # Run Blender with your .blend file
    subprocess.Popen([
        blender_path,
        "-b",  # background mode
        blend_file,
        "--python-exit-code", "1"
    ])

if __name__ == "__main__":
    run_visualizer()
""")

    PyInstaller.__main__.run([
        'run_visualizer.py',
        '--onefile',
        '--name=visualizer',
        '--add-data=audio-visualizer.blend;.',  
        '--noconsole'
    ])

if __name__ == "__main__":
    create_visualizer_exe()