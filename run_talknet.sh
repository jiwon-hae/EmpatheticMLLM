#!/bin/bash

module load ffmpeg

VID_FOLDER="/project/msoleyma_1026/EmpatheticResponseGeneration/MELD.Raw/train_splits"
SAVE_FOLDER="/project/msoleyma_1026/EmpatheticResponseGeneration/FaceCrop"
VIDEO_NAME="dia0_utt0"

count=1
find "$VID_FOLDER" -type f -name "*.mp4" | sort | while read -r video; do
    videoName=$(basename "$video")
    videoName="${videoName%.mp4}"
    echo "Processing video" $count ":" $videoName 
    python cropFace.py --videoName="$videoName" --videoFolder="$VID_FOLDER" --savePath="$SAVE_FOLDER"

    ((count++))
    if [ "$count" -ge 30 ]; then
        break
    fi
done