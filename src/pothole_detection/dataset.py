import os
from roboflow import Roboflow

def download_dataset(api_key, workspace, project_name, version_number):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download("yolov8")
    return dataset.location