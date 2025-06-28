import kagglehub
import shutil
import os

# Download dataset from Kaggle
path = kagglehub.dataset_download("adityajn105/flickr8k")

# Define your target directory inside Image_Captioning/Data/
target_dir = "Data/"

# Make sure the folder exists
os.makedirs(target_dir, exist_ok=True)

# Move all downloaded files into target directory
for item in os.listdir(path):
    source = os.path.join(path, item)
    destination = os.path.join(target_dir, item)

    # Move or copy files/folders
    if os.path.isdir(source):
        shutil.move(source, destination)
    else:
        shutil.copy2(source, destination)

print("Dataset successfully moved to:", target_dir)
