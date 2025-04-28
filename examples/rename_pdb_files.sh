#!/bin/bash

# Define the directory to search for directories
SEARCH_DIR="/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/halide_salt_and_2D_interfaces/dyn_substrate_trajs"
#SEARCH_DIR="/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/halide_salt_and_2D_interfaces/fixed_substrate_trajs"

# Loop through all directories in the search directory
for dir in "$SEARCH_DIR"/*/; do
    echo $dir
    # Check if the directory name matches the pattern <name>_
    dir_name=$(basename "$dir")
    if [[ "$dir_name" =~ ^(.*)_$ ]]; then
        # Extract the base name
        base_name="${BASH_REMATCH[1]}"
        # Remove trailing slash from directory path
        dir="${dir%/}"
        # Rename the directory to <name>__
        mv "$dir" "${SEARCH_DIR}/${base_name}__"
        echo "Renamed $dir to ${SEARCH_DIR}/${base_name}__"
    fi
done
