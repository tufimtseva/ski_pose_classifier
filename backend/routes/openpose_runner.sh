#!/usr/bin/env bash

#cd "$(dirname "$0")/../../openpose" || { echo "Failed to change directory"; exit 1; }
cd "$(dirname "$0")/../../" || { echo "Failed to change directory"; exit 1; }
#echo "Current directory: $(pwd)"
#
#
#if [ $# -lt 2 ]; then
#    echo "Usage: $0 <image_dir> <write_json>"
#    exit 1
#fi

echo "Current directory: $(pwd)"

#IMAGE_DIR="../$1"
#WRITE_JSON="../$2"

IMAGE_DIR="$1"
WRITE_JSON="$2"

echo "Executing command:"
#echo "./openpose/build/examples/openpose/openpose.bin --image_dir \"$IMAGE_DIR\" --write_json \"$WRITE_JSON\" --number_people_max 1 --keypoint_scale 3 --render_pose 0 --display 0"
echo "./backend/bin/mac/silicon/openpose.bin --image_dir \"$IMAGE_DIR\" --write_json \"$WRITE_JSON\" --model_folder \"./openpose/models/\" --number_people_max 1 --keypoint_scale 3 --render_pose 0 --display 0"

#./build/examples/openpose/openpose.bin \
#    --image_dir "$IMAGE_DIR" \
#    --write_json "$WRITE_JSON" \
#    --number_people_max 1 --keypoint_scale 3



./backend/bin/mac/silicon/openpose.bin \
  --image_dir "$IMAGE_DIR" \
  --write_json "$WRITE_JSON" \
  --model_folder "./openpose/models/" \
  --number_people_max 1 --keypoint_scale 3


##!/usr/bin/env bash
#cd "$(dirname "$0")/../../openpose"
#./build/examples/openpose/openpose.bin \
#--image_dir "../$1" \
#--write_json "../$2" \
#--number_people_max 1 --keypoint_scale 3 --render_pose 0 --display 0