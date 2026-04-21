#this version of the script uses the same parameters as original batch inference script from Amanda's archive
#!/bin/bash

# Directory containing videos
VIDEO_DIR="Z:/2024-09_FoodDep_Amanda/videos/female/final/20240913/bottomView/temp"

# SLEAP model path
MODEL_PATH="Z:/SLEAP/models/250222_172042.multi_instance.n=954/training_config.json"

# Output directory for predictions
OUTPUT_DIR="Z:/2024-09_FoodDep_Amanda/SLEAP/female/20240913_predictions"
  
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
    --tracking.target_instance_count 2 \
    --tracking.similarity instance \
    --tracking.match greedy \
    --tracking.post_connect_single_breaks 2 \
    --tracking.clean_instance_count 2 \
    -o "$OUTPUT_PATH" \
    --verbosity rich \
    --no-empty-frames \
    --gpu auto
  
  echo "Processing of $VIDEO_NAME completed."
done

echo "All videos processed."
