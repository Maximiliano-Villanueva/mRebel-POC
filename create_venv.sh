#!/bin/bash
# example: create_venv.sh rebel_venv
# Check if a name was provided
if [[ -z $1 ]]; then
    echo "Error: No virtual environment name provided."
    echo "Usage: $0 <venv_name>"
    exit 1
fi

VENV_NAME=$1

create_virtual_environment() {
    echo "Creating virtual environment $VENV_NAME..."
    python3 -m venv "$VENV_NAME"
    
    if [[ $? -eq 0 ]]; then
        echo "Virtual environment $VENV_NAME created successfully."
    else
        echo "Failed to create virtual environment."
        exit 1
    fi
}

create_virtual_environment
