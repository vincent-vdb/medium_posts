import argparse
from dataclasses import dataclass, field
from glob import glob
import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid, draw_bounding_boxes
from torchvision import transforms
import torchvision
import tqdm
import wandb

from utils import Detect, MultiBoxLoss, od_collate_fn
from blazeface import BlazeFace


@dataclass
class ModelParameters:
    """Class with all the model parameters"""
    batch_size: int = 32
    lr: float = 0.001
    scheduler_type: str = 'ReduceLROnPlateau'
    lr_scheduler_patience: int = 10
    epochs: int = 100
    classes: list = field(default_factory=lambda: ['face'])
    image_size: int = 128
    detection_threshold: float = 0.5
    blazeface_channels: int = 32
    focal_loss: bool = False
    model_path: str = 'weights/blazeface128.pt'
    use_wandb: bool = False
    augmentation: dict = None


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, labels_path, image_size: int, augment: A.Compose = None):
        self.labels_path = labels_path
        self.labels = list(sorted(glob(f'{labels_path}/*/*')))
        self.labels = [x for x in self.labels if os.stat(x).st_size != 0]
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((image_size, image_size))
        ])
        self.image_size = image_size

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.labels[idx].replace('labels', 'images')[:-3] + 'jpg'
        img = plt.imread(img_path)
        rescale_output = self.resize_and_pad(img, self.image_size)
        img = rescale_output['image']
        annotations = pd.read_csv(self.labels[idx], header=None, sep=' ')
        labels = annotations.values[:, 0]
        yolo_bboxes = annotations.values[:, 1:]
        cx = yolo_bboxes[:, 0]
        cy = yolo_bboxes[:, 1]
        w = yolo_bboxes[:, 2]
        h = yolo_bboxes[:, 3]
        x1 = (cx - w / 2) * rescale_output['x_ratio'] + rescale_output['x_offset']
        x2 = (cx + w / 2) * rescale_output['x_ratio'] + rescale_output['x_offset']
        y1 = (cy - h / 2) * rescale_output['y_ratio'] + rescale_output['y_offset']
        y2 = (cy + h / 2) * rescale_output['y_ratio'] + rescale_output['y_offset']
        x1 = np.expand_dims(x1, 1)
        x2 = np.expand_dims(x2, 1)
        y1 = np.expand_dims(y1, 1)
        y2 = np.expand_dims(y2, 1)
        target = np.concatenate([x1, y1, x2, y2, labels.reshape(-1, 1)], axis=1)
        if self.augment is not None:
            augmented = self.augment(image=img, bboxes=target)
            img = augmented['image']
            target = np.array(augmented['bboxes'])
        return self.transform(img.copy()), np.clip(target, 0, 1)

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def resize_and_pad(img, target_size=128):
        if img.shape[0] > img.shape[1]:
            new_y = target_size
            new_x = int(target_size * img.shape[1] / img.shape[0])
        else:
            new_y = int(target_size * img.shape[0] / img.shape[1])
            new_x = target_size
        output_img = cv2.resize(img, (new_x, new_y))
        top = max(0, new_x - new_y) // 2
        bottom = target_size - new_y - top
        left = max(0, new_y - new_x) // 2
        right = target_size - new_x - left
        output_img = cv2.copyMakeBorder(
            output_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )
        # Compute labels values updates
        x_ratio = new_x / target_size
        y_ratio = new_y / target_size
        x_offset = left / target_size
        y_offset = top / target_size

        return {'image': output_img, 'x_ratio': x_ratio, 'x_offset': x_offset, 'y_ratio': y_ratio, 'y_offset': y_offset}


def compute_image_with_boxes_grid(postprocessor, preds, labels, dbox_list, images):
    # Compute postprocessing for valid
    detections = postprocessor.forward((preds[:, :, :4], preds[:, :, 4:], dbox_list))
    # Make a grid image with bounding boxes
    classes_names = ['face', 'hand']
    imgs_with_boxes = []
    for i in range(min(len(images), 32)):
        if len(detections[i, :, 0] > model_params.detection_threshold) > 0:
            filtered_dets = detections[i, detections[i, :, 0] > model_params.detection_threshold, :]
            classes = [classes_names[int(pred_class)] for pred_class in filtered_dets[:, -1]]
            # Draw predicted boxes
            img_with_boxes = draw_bounding_boxes(
                ((images[i] * 0.5 + 0.5) * 255).to(torch.uint8),
                filtered_dets[:, 1:-1] * images.shape[-1],
                classes,
                "red"
            )
            # Draw ground truth labels
            classes = [classes_names[int(gt_class)] for gt_class in labels[i][:, -1]]
            img_with_boxes = draw_bounding_boxes(
                img_with_boxes,
                labels[i][:, :4] * images.shape[-1],
                classes,
                "green"
            )
            imgs_with_boxes.append(img_with_boxes.unsqueeze(0).cpu())
        else:
            imgs_with_boxes.append(((images[i] * 0.5 + 0.5) * 255).to(torch.uint8).unsqueeze(0).cpu())
    imgs_with_boxes = torch.cat(imgs_with_boxes)
    grid_images = torchvision.utils.make_grid(imgs_with_boxes, nrow=8)

    return grid_images


def train_model(
        net,
        dataloaders_dict,
        criterion,
        optimizer,
        scheduler,
        model_params,
        device,
):
    net = net.to(device)
    if model_params.use_wandb:
        postprocessor = Detect()
        run = wandb.init(
            project='BlazeFace',
            config={
                'model_params': model_params,
            }
        )

    for epoch in range(model_params.epochs):
        curr_lr = scheduler.optimizer.param_groups[0]['lr']
        # Train
        running_loss = 0.
        running_loc_loss = 0.
        running_class_loss = 0.
        for images, targets in tqdm.tqdm(dataloaders_dict['train']):
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            loss_l, loss_c = criterion(outputs, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loc_loss += loss_l.item()
            running_class_loss += loss_c.item()
        if model_params.use_wandb:
            train_grid_images = compute_image_with_boxes_grid(postprocessor, outputs, targets, net.dbox_list, images)
        # Eval
        net.eval()
        val_loss = 0.
        val_loc_loss = 0.
        val_class_loss = 0.
        with torch.no_grad():
            for images, targets in dataloaders_dict['val']:
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]
                outputs = net(images)
                loss_l, loss_c = criterion(outputs, targets)
                loss = loss_l + loss_c
                val_loss += loss.item()
                val_loc_loss += loss_l.item()
                val_class_loss += loss_c.item()

        train_loss = running_loss / len(dataloaders_dict['train'])
        train_loc_loss = running_loc_loss / len(dataloaders_dict['train'])
        train_class_loss = running_class_loss / len(dataloaders_dict['train'])
        val_loss = val_loss / len(dataloaders_dict['val'])
        val_loc_loss = val_loc_loss / len(dataloaders_dict['val'])
        val_class_loss = val_class_loss / len(dataloaders_dict['val'])
        print(f'[{epoch + 1}] train loss: {train_loss:.3f} | val loss: {val_loss:.3f}')
        scheduler.step(val_loss)
        # Save model
        torch.save(net.state_dict(), model_params.model_path)

        if model_params.use_wandb:
            # Compute validation images with boxes
            val_grid_images = compute_image_with_boxes_grid(postprocessor, outputs, targets, net.dbox_list, images)
            # Log to W&B
            wandb.log({
                'train_results': wandb.Image(train_grid_images),
                'val_results': wandb.Image(val_grid_images),
                'train_loss': train_loss,
                'train_loc_loss': train_loc_loss,
                'train_class_loss': train_class_loss,
                'val_loss': val_loss,
                'val_loc_loss': val_loc_loss,
                'val_class_loss': val_class_loss,
                'lr': curr_lr,
            })
            # Log model
            run.log_model(name='blazeface', path=model_params.model_path)

    if model_params.use_wandb:
        run.finish()

if __name__ == '__main__':
    # Parse the args
    parser = argparse.ArgumentParser(description='Train blaze face model')
    parser.add_argument('--dataset', help='the dataset path', type=str, default='./datasets/')
    parser.add_argument('--wandb', help='use wandb (default to none)', action='store_true', default=False)
    parser.add_argument('--batch_size', help='the batch size', type=int, default=256)
    parser.add_argument('--epochs', help='the number of epochs', type=int, default=100)
    parser.add_argument('--lr', help='the initial learning rate', type=float, default=0.001)
    parser.add_argument('--det_threshold', help='the detection threshold', type=float, default=0.5)
    parser.add_argument('--img_size', help='the resized image size', type=int, default=128)
    parser.add_argument('--focal', help='use focal loss (default to none)', action='store_true', default=False)
    parser.add_argument('--channels', help='BlazeFace input channels', type=int, default=32)
    parser.add_argument('--original', help='Use original architecture', action='store_true', default=False)
    args = parser.parse_args()

    model_params = ModelParameters(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        image_size=args.img_size,
        detection_threshold=args.det_threshold,
        focal_loss=args.focal,
        use_wandb=args.wandb,
        blazeface_channels=args.channels,
    )

    augment = A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomCropFromBorders(
                crop_left=0.05,
                crop_right=0.05,
                crop_top=0.05,
                crop_bottom=0.05,
                p=0.9,
            ),
            A.Affine(
                rotate=(-30, 30),
                scale=(0.8, 1.1),
                keep_ratio=True,
                translate_percent=(-0.05, 0.05),
                cval=(128, 128, 128),
                p=0.9,
            ),
        ],
        bbox_params=A.BboxParams(format='albumentations')
    )
    model_params.augmentation = augment.to_dict()

    os.makedirs("weights", exist_ok=True)
    # Data loaders
    train_dataset = MyDataset(args.dataset + '/train/labels/', image_size=model_params.image_size, augment=augment)
    valid_dataset = MyDataset(args.dataset + '/validation/labels/', image_size=model_params.image_size)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=od_collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=od_collate_fn
    )
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
    # Model
    if model_params.image_size == 256:
        model = BlazeFace(back_model=True)
    else:
        model = BlazeFace()
    model.load_anchors('anchors.npy')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device, focal=model_params.focal_loss, dbox_list=model.dbox_list)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=model_params.lr_scheduler_patience)
    # Train the model
    train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        scheduler,
        model_params,
        device=device,
    )