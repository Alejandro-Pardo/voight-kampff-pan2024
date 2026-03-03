import os
import shutil

# Directory where the files are located
def organize_files():
    source_dir = 'data/machines'

# List all files in the source directory
    files = os.listdir(source_dir)

# Loop through each file
    for file in files:
    # Create a directory with the same name as the file (without the extension)
        dir_name = os.path.splitext(file)[0]
        dir_name = os.path.join(source_dir, dir_name)
        os.makedirs(dir_name, exist_ok=True)

    # Move the file to the newly created directory
        shutil.move(os.path.join(source_dir, file), dir_name)

#organize_files()

import json
from pathlib import Path

def join_jsonl_files(input_files, output_file):
    """
    Join multiple JSONL files into a single JSONL file.
    
    Args:
        input_files (list): List of paths to input JSONL files
        output_file (str): Path to output JSONL file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for input_file in input_files:
            try:
                with open(input_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # Verify each line is valid JSON
                        try:
                            json.loads(line.strip())
                            outfile.write(line)
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line in {input_file}: {line.strip()}")
            except FileNotFoundError:
                print(f"Warning: Could not find file {input_file}")
            except Exception as e:
                print(f"Error processing file {input_file}: {str(e)}")


input_files = ['data/human_parsed.jsonl', 'data/machines/alpaca-7b/alpaca-7b_parsed_relabeled.jsonl']
output_file = 'combined.jsonl'

join_jsonl_files(input_files, output_file)
print(f"Successfully joined files into {output_file}")