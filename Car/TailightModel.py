import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.optim as optim



model_path = input("Enter the path to the pre-trained model ")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

    

train_data = datasets.ImageFolder(
    root="./Car/Data", 
    transform=transform
    )

test_data  = datasets.ImageFolder(
    root="./Car/Data", 
    transform=transform
    )

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader  = DataLoader(test_data, batch_size=32, shuffle=True)



num_classes = len(train_data.classes)

class_to_index = train_data.class_to_idx
index_to_class = {index: class_name for class_name, index in class_to_index.items()}


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

learning_rate = 1e-2
epochs = 10
batch_size = 32

def train_loop(dataloader, model, loss_fn, optimizer):
    total_loss, num_batches = 0, 0
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        num_batches += 1
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss / num_batches

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class, filepath, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model

# Check if a pre-trained model exists


if os.path.exists(model_path):
    print(f"Found existing model at {model_path}")
    choice = input("Do you want to (1) load existing model or (2) train new model (3) train existing model Enter 1 or 2 or 3: ")
    
    if choice == '1':
        # Load existing model
        model = load_model(CNN, model_path, device)
        print("Loaded pre-trained model. Testing performance...")
        
        loss_fn = nn.CrossEntropyLoss().to(device)
        test_loop(test_dataloader, model, loss_fn)
        
    elif choice == '2':
        loss=[]
        iteration = []
        epochs = int(input("Enter number of epochs to train the existing model: "))
        learning_rate = input("Enter learning rate for training the existing model: ")
        learning_rate = float(learning_rate) if learning_rate else 5e-5
        # Train existing model
        print("Training existing model...")
        print(learning_rate)
        model = CNN().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss.append(train_loop(train_dataloader, model, loss_fn, optimizer))
            test_loop(test_dataloader, model, loss_fn)
            iteration.append(t+1)
        print("Training Done!")
        save_model(model, model_path)
        # Fixed version:
        plt.figure(figsize=(10, 6))
        plt.plot(iteration, loss, marker='o', color='blue', label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()  # This is important!
    elif choice == '3':
        loss=[]
        iteration = []
        epochs = int(input("Enter number of epochs to train the existing model: "))
        learning_rate = input("Enter learning rate for training the existing model: ")
        learning_rate = float(learning_rate) if learning_rate else 5e-5
        # Train existing model
        print("Training existing model...")
        print(learning_rate)
        model = load_model(CNN, model_path, device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss.append(train_loop(train_dataloader, model, loss_fn, optimizer))
            test_loop(test_dataloader, model, loss_fn)
            iteration.append(t+1)
        print("Training Done!")
        save_model(model, model_path)
        # Fixed version:
        plt.figure(figsize=(10, 6))
        plt.plot(iteration, loss, marker='o', color='blue', label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()  # This is important!
else:
    loss=[]
    iteration = []
    epochs = int(input("Enter number of epochs to train the existing model: "))
    learning_rate = input("Enter learning rate for training the existing model: ")
    learning_rate = float(learning_rate) if learning_rate else 5e-5
    # Train existing model
    print("Training existing model...")
    print(learning_rate)
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss.append(train_loop(train_dataloader, model, loss_fn, optimizer))
        test_loop(test_dataloader, model, loss_fn)
        iteration.append(t+1)
    print("Training Done!")
    save_model(model, model_path)
    # Fixed version:
    plt.figure(figsize=(10, 6))
    plt.plot(iteration, loss, marker='o', color='blue', label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()  # This is important!