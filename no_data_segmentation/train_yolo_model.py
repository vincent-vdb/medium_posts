import argparse

from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO instance segmentation model")
    parser.add_argument(
        "--epochs", type=int, help="number of training epochs", default=10
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="path to dataset.yaml file",
        default="datasets/animals/dataset.yaml",
    )
    parser.add_argument(
        "--input_model", type=str, help="input model for YOLO", default="yolov8n-seg.pt"
    )
    args = parser.parse_args()

    model = YOLO(args.input_model)
    results = model.train(data=args.dataset_path, epochs=args.epochs)
