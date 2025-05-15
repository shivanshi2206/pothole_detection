import os
from ultralytics import YOLO
from pothole_detection.utils import load_config
from roboflow import Roboflow

def main():
    cfg = load_config()

    # === Roboflow Dataset Download ===
    rf = Roboflow(api_key=cfg["roboflow"]["api_key"])
    project = rf.workspace(cfg["roboflow"]["workspace"]).project(cfg["roboflow"]["project"])
    version = project.version(cfg["roboflow"]["version"])
    dataset = version.download("yolov8")
    yaml_path = os.path.join(dataset.location, "data.yaml")

    # === Model Training ===
    model = YOLO(cfg["training"]["model"])
    results = model.train(
        data=yaml_path,
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["training"]["imgsz"],
        patience=cfg["training"]["patience"],
        batch=cfg["training"]["batch"],
        optimizer=cfg["training"]["optimizer"],
        lr0=cfg["training"]["lr0"],
        lrf=cfg["training"]["lrf"],
        device=cfg["training"]["device"],
        seed=cfg["training"]["seed"],
        mosaic=cfg["training"]["mosaic"],
        cos_lr=cfg["training"]["cos_lr"],
        cache=cfg["training"]["cache"]
    )

    # Save model
    model_path = f"saved_models/yolov8s_seg_e{cfg['training']['epochs']}.pt"
    model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
