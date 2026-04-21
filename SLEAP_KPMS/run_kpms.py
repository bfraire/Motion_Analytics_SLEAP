import keypoint_moseq as kpms
import os
import glob

print(f"Keypoint-Moseq version: {kpms.__version__}")


h5_folder = r"Z:\2024-09_FoodDep_Amanda\SLEAP\female\h5_extracted"

if os.path.exists(h5_folder):
    print("Folder exists")
else:
    print("Folder not found. Check the path.")

files = os.listdir(h5_folder)
print("Files in folder:", files)

h5_files = glob.glob(h5_folder + "\\*analysis.h5")

print(f"Found {len(h5_files)} HDF5 files to process")

all_data = [kpms.io.load_data(f) for f in h5_files]

merged_data = kpms.preprocessing.merge_datasets(all_data)

kpms.io.save_data(merged_data, h5_folder + "\\merged_keypoints.h5")

model = kpms.models.fit_hmm(merged_data)
