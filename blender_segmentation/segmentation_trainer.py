import argparse
import os
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(glob(image_dir + '/image*.jpg'))
        self.masks = [img.replace('image','mask') for img in self.images]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = plt.imread(self.images[idx])
        mask = plt.imread(self.masks[idx])[:,:,0]
        if image.shape[:2] != mask.shape[:2]:
            print('shape mismatch mask and image', self.images[idx])
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        mask = (mask > 128).to(torch.float)

        return image, mask


def get_train_transform():
    return A.Compose([
        A.Rotate(limit=(-45, 45)),
        A.RandomCropFromBorders(crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1, p=0.9),
        A.Resize(256, 256),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_valid_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders
    train_dataset = CustomDataset(os.path.join(args.dataset, 'train'), transform=get_train_transform())
    valid_dataset = CustomDataset(os.path.join(args.dataset, 'valid'), transform=get_valid_transform())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid',
    )
    model = model.to(device)

    # Define loss function and optimizer
    if args.loss.lower() == 'BCE':
        criterion = nn.BCELoss()
    else:
        criterion = smp.losses.FocalLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, args.num_epochs, device)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_model.pth')
    print("Model saved as 'trained_model.pth'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to the dataset directory")
    parser.add_argument("--encoder", type=str, default="timm-mobilenetv3_large_100", help="Encoder architecture")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--loss", type=str, default="focal", help="Loss, either BCE or focal")


    args = parser.parse_args()
    main(args)