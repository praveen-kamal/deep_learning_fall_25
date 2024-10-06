# test.py
import torch
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


# Load the pre-trained model
def load_model(model_path, configs, num_layers, num_hidden, num_classes=36):
    model = RNN(
        num_layers=num_layers,
        num_hidden=num_hidden,
        configs=configs,
        num_classes=num_classes,
    )
    model.load_state_dict(torch.load(model_path))
    return model


# Evaluate the model on the test dataset
def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)  # Total samples
            correct += (predicted == labels).sum().item()  # Correct predictions

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    # Dataset paths
    test_data_path = "./asl_dataset_test"

    # Hyperparameters (use same values as during training)
    num_layers = 4
    num_hidden = [32, 32, 32, 32]  # Example hidden layers
    num_classes = 36  # 26 letters + 10 digits

    # Load the test dataset
    test_dataset = ASLDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the model
    configs = Configs()
    model_path = "model_epoch_20.pth"  # Path to the pre-trained model
    model = load_model(model_path, configs, num_layers, num_hidden, num_classes)

    # Check if GPU is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model
    evaluate_model(model, test_loader, device)
