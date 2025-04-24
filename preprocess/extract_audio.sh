#!/bin/bash

# This script extracts .wav audio from all .mp4 files in the specified directory and stores them in the specified output directory.

# Get the input and output directories from the arguments
INPUT_DIRECTORY="$1"
OUTPUT_DIRECTORY="$2"

# Check if the specified input directory exists
if [ ! -d "$INPUT_DIRECTORY" ]; then
  echo "Input directory $INPUT_DIRECTORY does not exist."
  exit 1
fi

# Check if the specified output directory exists; if not, create it
if [ ! -d "$OUTPUT_DIRECTORY" ]; then
  echo "Output directory $OUTPUT_DIRECTORY does not exist. Creating it..."
  mkdir -p "$OUTPUT_DIRECTORY"
fi

# Change to the specified input directory
cd "$INPUT_DIRECTORY"

# Find all .mp4 files in the directory
mp4_files=(*.mp4)

# Loop over each .mp4 file and extract the .wav audio
for mp4_file in "${mp4_files[@]}"; do
  # Create an output filename by replacing .mp4 with .wav in the output directory
  output_file="$OUTPUT_DIRECTORY/${mp4_file%.mp4}.wav"
  
  echo "Extracting audio from $mp4_file to $output_file..."
  
  # Use ffmpeg to extract the audio as .wav
  ffmpeg -i "$mp4_file" -vn -acodec pcm_s16le "$output_file"

  # Check if ffmpeg command was successful
  if [ $? -eq 0 ]; then
    echo "Successfully extracted: $output_file"
  else
    echo "Failed to extract audio from $mp4_file"
  fi
done
