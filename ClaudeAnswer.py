import os
import torch
from torch import nn
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# Your model class (same as before)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def load_model(model_class, filepath, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model

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
        transforms.ToTensor(),                        # Convert to tensor (0-1 range)
    ])
    
    # Load and transform the image
    image = Image.open(image_path)
    image_tensor = transform(image)
    
    # Add batch dimension: [1, 1, 28, 28]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image

def predict_image(model, image_tensor, device):
    """Make prediction on a single image"""
    # FashionMNIST class names
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    # Move to device and get prediction
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities).item()
    
    predicted_label = class_names[predicted_class.item()]
    
    return predicted_label, confidence, probabilities

def visualize_prediction(original_image, image_tensor, predicted_label, confidence, probabilities):
    """Visualize the image and prediction results"""
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show original image
    ax1.imshow(original_image)
    ax1.set_title(f'Original Image')
    ax1.axis('off')
    
    # Show preprocessed image
    processed_img = image_tensor.squeeze().cpu().numpy()
    ax2.imshow(processed_img, cmap='gray')
    ax2.set_title(f'Processed (28x28)\nPredicted: {predicted_label}\nConfidence: {confidence:.2%}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show top 3 predictions
    probs = probabilities.squeeze().cpu().numpy()
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    print("\nTop 3 Predictions:")
    for i, idx in enumerate(top_3_indices):
        print(f"{i+1}. {class_names[idx]}: {probs[idx]:.2%}")

# Main execution
def test_custom_image(image_path):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = 'fashion_mnist_model.pth'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first.")
        return
    
    # Load model
    print("Loading trained model...")
    model = load_model(NeuralNetwork, model_path, device)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Preprocess image
    print("Preprocessing image...")
    image_tensor, original_image = preprocess_image(image_path)
    
    # Make prediction
    print("Making prediction...")
    predicted_label, confidence, probabilities = predict_image(model, image_tensor, device)
    
    # Show results
    print(f"\nPrediction: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    
    # Visualize (optional - comment out if you don't want plots)
    visualize_prediction(original_image, image_tensor, predicted_label, confidence, probabilities)

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "my_test_image.jpg"  # Change this to your image file
    
    # Test the image
    test_custom_image(image_path)
    
    # Or test multiple images
    # test_images = ["shirt.jpg", "shoe.png", "dress.jpg"]
    # for img_path in test_images:
    #     print(f"\n{'='*50}")
    #     print(f"Testing: {img_path}")
    #     print('='*50)
    #     test_custom_image(img_path)