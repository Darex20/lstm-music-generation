import os
import shutil

# Define the source directory where the POP909 dataset is stored
# Replace this with the actual path to the POP909 dataset on your local machine
source_dir = './POP909-Dataset'

# Define the target directory where you want to gather all `.mid` files
# This can be a directory like 'POP909_mid_files' in your desired location
target_dir = './extracted-POP909'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Initialize a counter for the moved files
moved_files_count = 0

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Check if the file is a .mid file
        if file.endswith('.mid') and not "v" in file:
            # Create the full path to the file
            file_path = os.path.join(root, file)
            # Create the target file path
            target_file_path = os.path.join(target_dir, file)
            
            # If a file with the same name exists, append a number to avoid overwrites
            if os.path.exists(target_file_path):
                file_name, file_extension = os.path.splitext(file)
                i = 1
                # Find a file name that doesn't exist yet
                while os.path.exists(os.path.join(target_dir, f"{file_name}_{i}{file_extension}")):
                    i += 1
                target_file_path = os.path.join(target_dir, f"{file_name}_{i}{file_extension}")
            
            # Copy the file to the target directory
            shutil.copy(file_path, target_file_path)
            # Increment the moved files count
            moved_files_count += 1

# Print the result
print(f"Moved {moved_files_count} .mid files to {target_dir}")
