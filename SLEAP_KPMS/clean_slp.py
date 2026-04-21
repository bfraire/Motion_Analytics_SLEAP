import sleap
from sleap import Labels  # Correct import for Labels
import os 

# Load labels using the correct method for SLEAP 1.4.1
filename = "Z:\CAP_SLEAPTrain_merged_v11_topdown.slp"
labels = sleap.load_file(filename)

# Load the video and get total frame count
import cv2
video_path = "2024-09_FoodDep_Amanda/videos/female/final/20240919/bottomView/20240919_FoodDep_F_postCAP_2110_2111_bottom.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Filter out frames that don't exist
valid_labeled_frames = [
    lf for lf in labels.labeled_frames if lf.frame_idx < total_frames
]

# Create a new Labels object
clean_labels = Labels(labeled_frames=valid_labeled_frames, videos=labels.videos)

# Save the cleaned file
print("Saving to:", os.getcwd())
clean_labels.save("cleaned_labels.slp")
