import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


labels_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

def load_model(model_class, filepath, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model


model_path = input("Enter the path to the pre-trained model ")
model = load_model(NeuralNetwork, model_path, device)

def preprocess_image(image_path):
    """
    Preprocess a custom image to match FashionMNIST format
    - Convert to grayscale
    - Resize to 28x28
    - Convert to tensor
    - Normalize like FashionMNIST (0-1 range)
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),                  # Resize to 28x28
        transforms.ToTensor(),# Convert to tensor (0-1 range)
        transforms.Lambda(lambda x: 1.0 - x)  # Normalize to [0, 1]
    ])

    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  
    return img

imagePath = input("Enter the path to the image you want to test: ")
img = preprocess_image(imagePath)  # Replace with your image path
plt.imshow(img.squeeze(), cmap="gray")
plt.show()
logits = model(img.to(device))
pred_probab = nn.Softmax(dim=1)(logits)
print(logits)
print(pred_probab)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
print(f"predicted class name: {labels_map[y_pred.item()]}")
