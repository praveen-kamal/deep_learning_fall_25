import os
import re

# Define the dataset path
dataset_dir = "./asl_dataset_test/"

# Regular expression to match the filenames and extract label and frame number
# Handles variations like hand1, hand2, bot, dif, seg_X, etc.
file_pattern = re.compile(r"hand\d+_([a-zA-Z0-9]+)_\w+_seg_(\d+)_cropped\.(\w+)")

# Loop through each directory (symbol) in the dataset
for symbol_dir in os.listdir(dataset_dir):
    symbol_path = os.path.join(dataset_dir, symbol_dir)

    if os.path.isdir(symbol_path):  # Make sure it's a directory
        sequence_count = 1  # Initialize sequence count for each symbol

        for filename in sorted(os.listdir(symbol_path)):
            # Match the filename pattern
            match = file_pattern.match(filename)

            if match:
                label = match.group(1)  # ASL symbol (0-9, a-z)
                frame_num = match.group(2)  # Frame number (1-5)
                ext = match.group(3)  # File extension (png, jpg, etc.)

                # Create new filename based on the sequence and frame number
                new_filename = f"{symbol_dir}_{sequence_count}_frame_{frame_num}.{ext}"

                # Full old and new file paths
                old_file_path = os.path.join(symbol_path, filename)
                new_file_path = os.path.join(symbol_path, new_filename)

                # Rename the file
                os.rename(old_file_path, new_file_path)

                # Print renaming action (optional, for logging)
                print(f"Renamed {old_file_path} to {new_file_path}")

                if int(frame_num) == 5:  # Every 5 frames, increment the sequence count
                    sequence_count += 1
