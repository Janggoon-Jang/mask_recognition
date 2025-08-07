import os
from pathlib import Path
from ultralytics import YOLO

def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    data_yaml = "/kaggle/input/dataset-final/data.yaml"
    model_path = "/kaggle/working/mask_recognition/src/MaskRecog/model/baseline.pt"

    model = YOLO(model_path)

    model.train(
        data=data_yaml,
        epochs=300,
        imgsz=640,
        batch=32,
        name="selected_dataset",
        project="results/runs",
        save=True,  
        plots=True, 
        verbose=True
    )

if __name__ == "__main__":
    main()