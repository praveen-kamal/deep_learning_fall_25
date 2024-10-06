# asl_dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ASLDataset(Dataset):
    def __init__(self, dataset_path, transform=None, seq_len=5):
        self.dataset_path = dataset_path
        self.transform = (
            transform if transform is not None else self._default_transforms()
        )
        self.seq_len = seq_len
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        # Assuming each subdirectory in dataset_path represents a class label
        for label in os.listdir(self.dataset_path):
            label_dir = os.path.join(self.dataset_path, label)
            if os.path.isdir(label_dir):
                # Sort frames in sequence order
                frame_files = sorted(os.listdir(label_dir))
                for i in range(0, len(frame_files) - self.seq_len + 1, self.seq_len):
                    sequence = frame_files[
                        i : i + self.seq_len
                    ]  # Get a sequence of frames
                    self.data.append(
                        [os.path.join(label_dir, frame) for frame in sequence]
                    )
                    self.labels.append(
                        int(label) if label.isdigit() else ord(label) - ord("a") + 10
                    )  # Mapping letters and digits

    def _default_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    (64, 64)
                ),  # Resize all images to 64x64, or adjust as per your dataset
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1, 1] range
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self.data[idx]
        sequence = []
        for frame_path in frames:
            image = Image.open(frame_path).convert("RGB")  # Convert to RGB
            if self.transform:
                image = self.transform(image)
            sequence.append(image)

        sequence = torch.stack(
            sequence
        )  # Stack frames along a new dimension (seq_len, channels, height, width)
        label = self.labels[idx]
        return sequence, label
