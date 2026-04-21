#this version of the script uses the same parameters as original batch inference script from Amanda's archive
#!/bin/bash

# Directory containing videos
VIDEO_DIR="Z:/2025-01_FoodDepM_Amanda/videos/final/bottomView"

# SLEAP model path
MODEL_PATH="Z:/SLEAP/models/250222_172042.multi_instance.n=954_bottomup/training_config.json"

# Output directory for predictions
OUTPUT_DIR="Z:/2025-01_FoodDepM_Amanda/SLEAP/predictions_v1"

# Iterate over each video in the directory
for VIDEO_PATH in "$VIDEO_DIR"/*.mp4; do
  # Extract video filename without extension
  VIDEO_NAME=$(basename "$VIDEO_PATH" .mp4)
  
  # Define output path
  OUTPUT_PATH="$OUTPUT_DIR/${VIDEO_NAME}.predictions.slp"
  
  # Run SLEAP inference
  sleap-track "$VIDEO_PATH" \
    -m "$MODEL_PATH" \
    --tracking.tracker flowmaxtracks \
    --max_instances 2 \
    --tracking.max_tracks 2 \
    --tracking.target_instance_count 2 \
    --tracking.similarity instance \
    --tracking.match greedy \
    --tracking.track_window 2 \
    --tracking.pre_cull_to_target 2 \
    --tracking.pre_cull_iou_threshold 0.9 \
    --tracking.max_tracking 1 \
    -o "$OUTPUT_PATH" \
    --verbosity json \
    --no-empty-frames \
    --gpu auto
  
  echo "Processing of $VIDEO_NAME completed."
done

echo "All videos processed."
