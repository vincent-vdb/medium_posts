import argparse

from ultralytics import YOLO


def train_yolo_model(input_model: str, dataset_filepath: str, epochs: int):
    model = YOLO(input_model)
    model.train(data=dataset_filepath, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO instance segmentation model")
    parser.add_argument("--epochs", type=int, help="number of training epochs", default=20)
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="path to dataset.yaml file",
        default="datasets/animals/dataset.yaml",
    )
    parser.add_argument("--input_model", type=str, help="input model for YOLO", default="yolov8n-seg.pt")
    args = parser.parse_args()

    train_yolo_model(args.input_model, args.dataset_path, args.epochs)
