import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from collections import Counter
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# comment

torch.manual_seed(18)
random.seed(18)
np.random.seed(18)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(18)

transform = transforms.ToTensor()


# Define transformations (convert to tensor + normalize if you want)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),  # Convert PIL image to Tensor
    # NORMALISATION -do or not do- 3 channels with each entry in range [0-1]
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) 
])

# Load datasets
train_dataset = datasets.ImageFolder(root="train", transform=transform)
test_dataset = datasets.ImageFolder(root="test", transform=transform)

# reads an example image
image, label = train_dataset[84]
print(image, label)

# Use DataLoaders -> adjust batch size (batch size-number of pictures processed at once)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoodCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #all the convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 5, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 5, 1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 5, 1, padding=1)

        #pooling reduces the dimensions by half
        self.pool = nn.MaxPool2d(2,2)

        self.flatten = nn.Flatten()

        #fully connected layers of the model, working with the flatten version of the input
        self.fc1 = nn.Linear(9216, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128 ,91)

        #optionally
        #self.fc1 = Linear(64*32*32, 256)
        #self.fc2 = Linear(256, 91)

        self._train_and_save_model()

    def forward(self, X):
        #convulution and pooling operations
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        X = F.relu(self.conv2(X))
        X = self.pool(X)
        X = F.relu(self.conv3(X))
        X = self.pool(X)
        X = F.relu(self.conv4(X))
        X = self.pool(X)
        X = F.relu(self.conv5(X))
        X = self.pool(X)

        #flattens the input to fully connected layers
        X = self.flatten(X)

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        #last layer categorises the input
        X = self.fc3(X)
        return X
    
    def _train_and_save_model(self):
        pass

def _train_and_save_model(self):
    self.to(device)
    torch.manual_seed(18)
    epochs = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    best_train_accuracy = 0.0

    for e in range(epochs):
        self.train()
        training_correct = 0
        total_train = 0

        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)

            #resets the optimizer to 0
            optimizer.zero_grad()
            #outputs a tensor with values assigned to each class
            y_pred = self(X_train)
            loss = criterion(y_pred, y_train)

            #backpropagates; accumulates the gradient for each parameter
            loss.backward()
            #parameters update
            optimizer.step()

            label_pred = torch.argmax(y_pred, dim=1)
            #total count of correct classifications
            training_correct += (label_pred == y_train).sum().item()
            #number of label in an epoch
            total_train += y_train.size(0)

        train_accuracy = (training_correct / total_train) / 100
        print(f"Training accuracy of epoch {e+1}: {train_accuracy}.")

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            torch.save(self.state_dict(), "best_model.pth")

FoodCNN._train_and_save_model = _train_and_save_model

# Load the best model weights
model = FoodCNN().to(device)
model.load_state_dict(torch.load("best_model.pth"))

def calculate_test_accuracy(model):
    model.eval()
    test_correct = 0
    total_test = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            predictions = torch.argmax(outputs, dim=1)
            test_correct += (predictions == y_test).sum().item()
            total_test += y_test.size(0)

    test_accuracy = (test_correct / total_test) * 100
    return test_accuracy

final_test_acc = calculate_test_accuracy(model)
print(f"Final Test Accuracy: {final_test_acc:.2f}%")

def user_behaviour_simulation(model):
    model.eval()

    random_indices = random.sample(range(len(test_dataset)), 10)

    true_labels = []
    predicted_labels = []

    for idx in random_indices:
        image, label = test_dataset[idx]
        true_labels.append(label)

        #adjusts an image to be viewed as a batch
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            pred_label = torch.argmax(output, dim=1).item()
            predicted_labels.append(pred_label)

    #maps the indices of the classes to their real names from the dataset
    class_names = test_dataset.classes
    true_class_names = [class_names[i] for i in true_labels]
    pred_class_names = [class_names[i] for i in predicted_labels]

    #stores the classes names with their counts in a separate dictionaries
    true_counts = dict(Counter(true_class_names))
    pred_counts = dict(Counter(pred_class_names))

    #formats output as expected
    print("Predicted Class Frequencies:", pred_counts)
    print("True Class Frequencies:", true_counts)

user_behaviour_simulation(model)