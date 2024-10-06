# demo.py
import torch
from PIL import Image
from torchvision import transforms
from rnn_model import RNN  # Import the model
from spatiotemporal_lstm_cell import SpatioTemporalLSTMCell


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
    model.eval()  # Set model to evaluation mode
    return model


# Transform function for resizing, converting to tensor, and normalizing the input
def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),  # Resize images to 64x64
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize the image
        ]
    )


# Load a sequence of frames from a directory
def load_sequence(sequence_dir, transform, seq_len=5):
    # Load all frames in the directory, sorted by name
    frames = sorted(
        [f for f in os.listdir(sequence_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    )
    sequence = []
    for frame in frames[:seq_len]:  # Take the first `seq_len` frames
        img_path = os.path.join(sequence_dir, frame)
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        image = transform(image)  # Apply the transformations
        sequence.append(image)

    # Stack the sequence into a single tensor
    return torch.stack(sequence).unsqueeze(
        0
    )  # Add a batch dimension (1, seq_len, channels, height, width)


# Perform inference and return the predicted label
def predict(model, sequence, device):
    sequence = sequence.to(device)  # Move sequence to device (GPU/CPU)
    outputs = model(sequence)  # Forward pass
    _, predicted = torch.max(outputs.data, 1)  # Get the predicted class index
    return predicted.item()  # Return the predicted label


# Mapping from class index to ASL symbol (0-9, a-z)
def get_label_name(label_idx):
    if 0 <= label_idx <= 9:
        return str(label_idx)  # Digits 0-9
    else:
        return chr(label_idx - 10 + ord("a"))  # Letters a-z


if __name__ == "__main__":
    import os

    # Configuration and paths
    model_path = "model_epoch_20.pth"  # Path to the pre-trained model
    sequence_dir = "./test_seq/q"  # Directory containing the sequence of frames

    # Model parameters (same as used during training)
    num_layers = 4
    num_hidden = [32, 32, 32, 32]  # Example hidden layers
    num_classes = 36  # 26 letters + 10 digits

    # Load the model
    configs = Configs()
    model = load_model(model_path, configs, num_layers, num_hidden, num_classes)

    # Set up device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and transform the sequence of frames
    transform = get_transform()
    sequence = load_sequence(sequence_dir, transform)

    # Make a prediction
    predicted_label_idx = predict(model, sequence, device)
    predicted_label = get_label_name(predicted_label_idx)

    # Display the predicted ASL symbol
    print(f"Predicted ASL symbol: {predicted_label}")
