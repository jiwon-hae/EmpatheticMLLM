#!/bin/bash
module load ffmpeg

VID_FOLDER="/project/msoleyma_1026/EmpatheticResponseGeneration/MELD.Raw/output_repeated_splits_test_fixed"
SAVE_FOLDER="/project/msoleyma_1026/EmpatheticResponseGeneration/FaceCrop/output_repeated_splits_test"

count=1
find "$VID_FOLDER" -type f -name "*.mp4" ! -name "._*" | sort | while read -r video; do
    videoName=$(basename "$video")
    videoName="${videoName%.mp4}"

    if [ -d "$SAVE_FOLDER/$videoName/pyavi/highest_faces_encoded.mp4" ]; then
        echo "Skipping $videoName (output already exists)"
        ((count++))
        continue
    fi
    
    echo "Processing video" $count ":" $videoName 
    python cropFace.py --videoName="$videoName" --videoFolder="$VID_FOLDER" --savePath="$SAVE_FOLDER"
    ((count++))
done
echo 'DONE'