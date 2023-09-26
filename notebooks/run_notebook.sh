#!/bin/bash

# Check the number of arguments for just the file
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <notebook-file> [resolution] [sim_id]"
    exit 1
fi

# Check if the file exists
if [ ! -f "$1" ]; then
    echo "File '$1' does not exist."
    exit 1
fi

# Assign the first argument to notebook_file variable
notebook_file="$1"

# Convert the notebook to a Python script
jupyter nbconvert --to script "$notebook_file"

# Extract the base name (without extension) from the file
base_name=$(basename "$notebook_file" .ipynb)
script_name="${base_name}.py"

# Modify the script to check command-line arguments for resolution and sim_id
echo "import sys" > "/tmp/${script_name}"
if [ "$#" -ge 2 ]; then
    sed -i 's/^\(resolution\s*=\)/# \1/' "${script_name}" # Comment out the original assignments in the generated Python script
    echo "resolution = sys.argv[1]" >> "/tmp/${script_name}"
fi
if [ "$#" -ge 3 ]; then
    sed -i 's/^\(sim_id\s*=\)/# \1/' "${script_name}" # Comment out the original assignments in the generated Python script
    echo "sim_id = sys.argv[2]" >> "/tmp/${script_name}"
fi
cat "${script_name}" >> "/tmp/${script_name}"

# Create the out directory if it doesn't exist
mkdir -p out

# Use nohup to run the script in the background with provided arguments (if any) and redirect outputs and errors to an out file inside the out directory
nohup ipython "/tmp/${script_name}" "${@:2}" > "out/${base_name}.out" 2>&1 &

# Delete the original Python script
rm "${script_name}"

# Notify user about the saved location
echo "Output and errors will be saved in out/${base_name}.out"

