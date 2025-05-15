import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "pothole_detection"

list_of_files = [
    ".github/workflows/.gitkeep",
    
    # Source files
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/dataset.py",
    f"src/{project_name}/train.py",
    f"src/{project_name}/visualize.py",
    f"src/{project_name}/utils.py",
    
    # Configs
    "configs/config.yaml",
    "requirements.txt",
    
    # Outputs
    "saved_models/.gitkeep",
    "results/.gitkeep",
    
    # Colab or dev notebooks
    "notebooks/training.ipynb",

    # Optional project files
    "README.md",
    ".gitignore",
    "setup.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")