# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rnn_model import RNN  # Import the model
from asl_dataset import ASLDataset


# Configuration class for model hyperparameters and data attributes
class Configs:
    patch_size = 1
    img_channel = 3
    img_width = 64
    filter_size = 3
    stride = 1
    layer_norm = True


# Dataset paths
train_data_path = "./asl_dataset_train"

# Hyperparameters
num_epochs = 20
batch_size = 2
learning_rate = 0.1
num_layers = 4
num_hidden = [32, 32, 32, 32]  # Example hidden layers
num_classes = 36  # 26 letters + 10 digits

# Load Dataset
train_dataset = ASLDataset(train_data_path)  # Custom dataset for ASL
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define Model
configs = Configs()
model = RNN(
    num_layers=num_layers,
    num_hidden=num_hidden,
    configs=configs,
    num_classes=num_classes,
)

# Check if GPU is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    print(f"Epoch: {epoch}")

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs = model(inputs)  # Shape: (batch_size, num_classes)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
        )

    # Save the model after each epoch
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    print(f"Model saved for epoch {epoch + 1}")

print("Training Finished")
