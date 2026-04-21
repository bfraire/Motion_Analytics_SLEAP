import sleap_io as sio
import numpy as np

# === CONFIGURE ===
old_path = r"\\smith-nas.ucsd.edu\lab\SLEAP\experiments\bottom-up_multi-instance_16ptSkeleton_AVN\projects\16ptskeleton_arborized_v1_n=117frames.slp"
clean_skeleton_path = r"\\smith-nas.ucsd.edu\lab\SLEAP\experiments\bottom-up_multi-instance_16ptSkeleton_AVN\projects\10ptskeleton_arborized_skeletonOnly.slp"
output_path = r"\\smith-nas.ucsd.edu\lab\SLEAP\experiments\bottom-up_multi-instance_16ptSkeleton_AVN\projects\10ptskeleton_arborized_v1_n=117frames_CLEANED.slp"
# =================

# Load files

old_labels = sio.load_slp(old_path)
clean_labels = sio.load_slp(clean_skeleton_path)

old_skel = old_labels.skeletons[0]
new_skel = clean_labels.skeletons[0]

old_names = old_skel.node_names
new_names = new_skel.node_names

# Map new node names to old indices
name_to_old_index = {name: i for i, name in enumerate(old_names)}
keep_indices = [name_to_old_index[name] for name in new_names]

# Rebuild instances
for lf in old_labels:
    new_instances = []

    for inst in lf.instances:
        old_points = inst.points
        new_points = old_points[keep_indices]

        new_inst = sio.Instance(
            skeleton=new_skel,
            points=new_points,
            track=inst.track
        )

        new_instances.append(new_inst)

    lf.instances = new_instances

# Replace skeleton
old_labels.skeletons = [new_skel]

sio.save_slp(old_labels, output_path)

print("Saved rebuilt file:", output_path)