#!/bin/bash

# Usage: ./update_fermi_temp.sh <base-directory> <new-temperature>

BASE_DIR="$1"
NEW_TEMP="$2"

if [[ -z "$BASE_DIR" || -z "$NEW_TEMP" ]]; then
    echo "Usage: $0 <base-directory> <new-temperature>"
    exit 1
fi

# Loop over each subdirectory
for SUBDIR in "$BASE_DIR"/*/; do
    CONTROL_FILE="${SUBDIR}/control.in"

    if [[ -f "$CONTROL_FILE" ]]; then
        echo "Updating $CONTROL_FILE"
        # Use sed to replace the occupation_type line
        sed -i -E "s/^( *occupation_type +fermi +)[0-9.]+/\1${NEW_TEMP}/" "$CONTROL_FILE"
    fi
done