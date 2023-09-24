import os
import numpy as np
import os
import cv2
import math
import argparse
import time
import copy
import shutil

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--file_name')
args = parser.parse_args()

# def split_file(file_name):
#     # Read the contents from the text file
#     with open(f'{file_name}_clean.txt', 'r') as f:
#         lines = f.readlines()

#     # Create a dictionary to hold the split contents
#     split_contents = {}

#     # Process each line and split by the first integer value
#     for line in lines:
#         values = line.strip().split(',')
#         first_value = int(values[0])

#         if first_value not in split_contents:
#             split_contents[first_value] = []

#         split_contents[first_value].append(line)

#     # Save split contents into separate files
#     for key, content_list in split_contents.items():
#         file_name = f'{key:07d}.txt'

#         with open(file_name, 'w') as f:
#             f.writelines(content_list)

def stitch_files(file_name_val):
    # Get a list of files in the Colab files section
    files_in_colab = os.listdir()

    # Filter and sort files based on numeric part of filenames with leading zeros
    sorted_files = sorted([file_name for file_name in files_in_colab if file_name.endswith(".txt")], key=lambda x: int(x.split('.')[0]))

    combined_data = []

    # Loop through each sorted file
    for file_name in sorted_files:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                combined_data.append(line.strip())

    # Replace spaces with commas in the combined data and remove ".0" from the first column
    combined_data = [data_point.replace(' ', ',') for data_point in combined_data]
    combined_data = [data_point.replace('.0,', ',') if data_point.startswith('4.0') or data_point.startswith('5.0') or data_point.startswith('1.0') else data_point for data_point in combined_data]

    # Write the combined data to a new file
    combined_file_name = f'{file_name_val}.txt'

    with open(combined_file_name, 'w') as combined_file:
        for data_point in combined_data:
            combined_file.write(data_point + '\n')

    print("Combined data has been written to", combined_file_name)

if __name__ == "__main__":
    import glob
        
    if args.file_name:
        #os.mkdir(str(args.file_name))
        #shutil.copy(args.file_name + "_clean.txt", args.file_name)
        os.chdir(str(args.file_name))
        os.chdir("fisheye")
        os.chdir("300")
        os.chdir("labels")
        #os.mkdir("images")
        #shutil.copy(args.file_name + "_clean.txt", "labels")
        #os.chdir("labels")
        #split_file(str(args.file_name))
        stitch_files(str(args.file_name))
        os.chdir("../../../../")
        #stitch_files(str(args.file_name))

