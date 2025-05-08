import os
import itertools
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import random
from collections import Counter
import numpy as np
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(18)
random.seed(18)
np.random.seed(18)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(18)

# Define separate transforms for training and testing
train_transform = transforms.Compose([
    transforms.Resize(144),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Allow for fair testing, don't give model flipped images for example. 
test_transform = transforms.Compose([
    transforms.Resize(144),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets with correct transforms
train_dataset = datasets.ImageFolder(root="train", transform=train_transform) 
test_dataset = datasets.ImageFolder(root="test", transform=test_transform)

# Read an example image
image, label = train_dataset[84]
print(image.shape, label)

# Set num_workers 
num_workers = min(14, os.cpu_count())  # auto-detect cpu core count, cap at 6 (lower this cap if system crashes)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoodCNN(nn.Module):
    def __init__(self, plot_graph=True, early_stopping=True, patience=10, test_eval_interval=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # outputs (batch_size, 256, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.23),
            nn.Linear(128, 91)
        )
        
        # control flags
        self.plot_graph = plot_graph
        self.early_stopping = early_stopping  # saves time is model stops improving
        self.patience = patience # threshold to stop training
        
        self.test_eval_interval = test_eval_interval # test model every n epochs for plotting
        
    def forward(self, X):
        X = self.features(X)
        X = self.classifier(X)
        return X

    def _train_and_save_model(self):
        pass  
    
    def plot_accuracies(self, train_accuracies, test_accuracies):
        """Plots training and testing accuracies after training, and saves plot."""
        
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"accuracy_plot_{timestamp}.png"
        save_path = os.path.join('plots', filename)
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label='Train Accuracy')
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='x', label='Test Accuracy')
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path)
        
        if self.plot_graph:
            plt.show()

def calculate_test_accuracy(model, test_loader): 
    model.eval()  # set model to evaluation mode
    test_correct = 0
    total_test = 0

    with torch.no_grad():  # disable gradient computation (faster)
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            
            with autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.bfloat16):
                outputs = model(X_test)
            
            predictions = torch.argmax(outputs, dim=1)
            test_correct += (predictions == y_test).sum().item()
            total_test += y_test.size(0)

    test_accuracy = (test_correct / total_test) * 100
    return test_accuracy

def _train_and_save_model(self):
    self.to(device)
    torch.manual_seed(18)
    epochs = 100  # or however many you want

    #ADAM
    #optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # learning rate gradaully decreases naturally (fight overfitting)
    #SGD
    optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device=device_type)

    best_test_accuracy = 0.0
    train_accuracies = []
    test_accuracies = []
    patience_counter = 0  # for early stopping

    print("Device being used:", device)
    start_time = time.time()

    for e in range(epochs):
        self.train()
        training_correct = 0
        total_train = 0

        for X_train, y_train in tqdm(train_loader, desc=f"Epoch {e+1}"):
            X_train, y_train = X_train.to(device), y_train.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device_type, dtype=torch.float16 if device_type=="cuda" else torch.bfloat16):
                y_pred = self(X_train)
                loss = criterion(y_pred, y_train)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            label_pred = torch.argmax(y_pred, dim=1)
            training_correct += (label_pred == y_train).sum().item()
            total_train += y_train.size(0)
            
        scheduler.step()

        train_accuracy = (training_correct / total_train) * 100
        print(f"Training accuracy of epoch {e+1}: {train_accuracy:.2f}%")
        train_accuracies.append(train_accuracy)


        # Evaluate test accuracy every 'self.test_eval_interval' epochs
        if (e + 1) % self.test_eval_interval == 0 or e == epochs - 1:
            test_accuracy = calculate_test_accuracy(self, test_loader)
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy of epoch {e+1}: {test_accuracy:.2f}%")

            # Save best model
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                torch.save(self.state_dict(), "best_model.pth")
                patience_counter = 0  # reset patience counter
            else:
                patience_counter += 1
            
            # Early stopping check
            if self.early_stopping and patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {e+1}.")
                break
        
        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    # Plot at the end
    self.plot_accuracies(train_accuracies, test_accuracies)

FoodCNN._train_and_save_model = _train_and_save_model

import gc
import torch

gc.collect()               # Python garbage collector: clears unused CPU memory
torch.cuda.empty_cache() 

# Instantiate and train a new model
model = FoodCNN().to(device)
model._train_and_save_model()  