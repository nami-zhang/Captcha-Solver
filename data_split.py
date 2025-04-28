import os
import random
from shutil import copy2

# Directories
input_dir = "generated_captchas"  # Where your generated CAPTCHAs are stored
output_dir = "dataset"            # Target dataset directory
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Split ratios: 80% train, 10% val, 10% test
train_ratio = 0.8
val_ratio = 0.1  # validation percentage
# The remaining 10% is for test

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all CAPTCHA files
all_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
random.shuffle(all_files)
total_files = len(all_files)

# Compute split indices
train_end = int(total_files * train_ratio)
val_end = int(total_files * (train_ratio + val_ratio))

# Split the dataset
train_files = all_files[:train_end]
val_files = all_files[train_end:val_end]
test_files = all_files[val_end:]

# Function to copy files and write labels
def save_files_and_labels(file_list, target_dir):
    label_path = os.path.join(target_dir, "labels.txt")
    with open(label_path, "w") as label_file:
        for file_name in file_list:
            label = os.path.splitext(file_name)[0]  # Extract text from file name
            src_path = os.path.join(input_dir, file_name)
            dst_path = os.path.join(target_dir, file_name)
            copy2(src_path, dst_path)  # Copy file
            label_file.write(f"{file_name} {label}\n")  # Write label

# Process each dataset split
save_files_and_labels(train_files, train_dir)
save_files_and_labels(val_files, val_dir)
save_files_and_labels(test_files, test_dir)

print("Dataset prepared:")
print(f"  Training samples: {len(train_files)}")
print(f"  Validation samples: {len(val_files)}")
print(f"  Testing samples: {len(test_files)}")
