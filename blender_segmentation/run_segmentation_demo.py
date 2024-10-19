import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms

def preprocess_image(image):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to match the model's expected input size
    image_resized = cv2.resize(image_rgb, (256, 256))
    
    # Convert to PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    image_tensor = transform(image_resized)
    
    return image_tensor.unsqueeze(0)  # Add batch dimension

def run_demo():
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="timm-mobilenetv3_large_100",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid',
    )
    model.load_state_dict(torch.load('trained_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_tensor = preprocess_image(frame)
        input_tensor = input_tensor.to(device)

        # Run inference
        with torch.no_grad():
            mask = model(input_tensor)
            mask = mask.squeeze().cpu().numpy()
        # Resize mask to match the original frame size
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Create a colored overlay
        overlay = np.zeros_like(frame)
        overlay[mask_resized > 0.5] = [0, 255, 0]  # Green color for segmented areas

        # Blend the original frame with the overlay
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Display the result
        cv2.imshow('Hand segmentation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
