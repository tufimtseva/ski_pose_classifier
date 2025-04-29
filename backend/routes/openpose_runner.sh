#!/usr/bin/env bash

cd "$(dirname "$0")/../../" || { echo "Failed to change directory"; exit 1; }

echo "Current directory: $(pwd)"

IMAGE_DIR="$1"
WRITE_JSON="$2"

echo "Executing command:"
echo "./backend/bin/mac/silicon/openpose.bin --image_dir \"$IMAGE_DIR\" --write_json \"$WRITE_JSON\" --model_folder \"./openpose/models/\" --number_people_max 1 --keypoint_scale 3 --render_pose 0 --display 0"

./backend/bin/mac/silicon/openpose.bin \
  --image_dir "$IMAGE_DIR" \
  --write_json "$WRITE_JSON" \
  --model_folder "./openpose/models/" \
  --number_people_max 1 --keypoint_scale 3 \
  --render_pose 0 --display 0 --maximize_positives
