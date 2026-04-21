#this version of the script uses the same parameters as original batch inference script from Amanda's archive
#!/bin/bash

# Directory containing videos
VIDEO_DIR="Y:\behaviorAnalysis\SLEAP\202507_ModelTesting"

# SLEAP model path
MODEL_PATH="Z:/SLEAP/models/250222_172042.multi_instance.n=954_bottomup/training_config.json"

# Output directory for predictions
OUTPUT_DIR="Y:\behaviorAnalysis\SLEAP\202507_ModelTesting"
  
# Iterate over each video in the directory
for VIDEO_PATH in "$VIDEO_DIR"/*.mp4; do
  # Extract video filename without extensions
  VIDEO_NAME=$(basename "$VIDEO_PATH" .mp4)
  
  # Define output path
  OUTPUT_PATH="$OUTPUT_DIR/${VIDEO_NAME}.predictions.slp"
  
  # Run SLEAP inference
  sleap-track "$VIDEO_PATH" \
    -m "$MODEL_PATH" \
    --tracking.tracker flow \
    --max_instances 2 \
    --tracking.target_instance_count 2 \
    --tracking.similarity instance \
    --tracking.post_connect_single_breaks 1 \
    --tracking.clean_instance_count 2 \
    --tracking.clean_iou_threshold 0.9\
    --tracking.match greedy \
    -o "$OUTPUT_PATH" \
    --verbosity rich \
    --no-empty-frames \
    --open_in_gui True \
    --gpu auto
  
  echo "Processing of $VIDEO_NAME completed."
done

echo "All videos processed."
