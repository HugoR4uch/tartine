#!/bin/bash

# Define the source and target directories
SOURCE_DIR="/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/halide_salt_and_2D_interfaces/dyn_substrate_trajs"
TARGET_DIR="/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/halide_salt_and_2D_interfaces"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Loop through all directories in the source directory
for dir in "$SOURCE_DIR"/*/; do
    # Get the base name of the directory
    base_dir=$(basename "$dir")
    # Create a symlink in the target directory
    ln -sT "$dir" "$TARGET_DIR/$base_dir"
done

echo "Symlinks created in $TARGET_DIR"
