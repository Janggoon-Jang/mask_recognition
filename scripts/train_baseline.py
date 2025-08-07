import os
from pathlib import Path
from ultralytics import YOLO

def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    data_yaml = "/kaggle/input/mask-recog-dataset1-processed/final/data.yaml"
    model_path = "yolo11s.pt"

    model = YOLO(model_path)

    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name="baseline",
        project="results/runs",
        save=True,  
        plots=True, 
        verbose=True
    )

if __name__ == "__main__":
    main()