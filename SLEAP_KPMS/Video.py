import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import os


class Video:
    '''
    Load an h5 datafile containing data extracted from a SLEAP inference, 
    predicting pose position over entire video
    '''
    
    def __init__(self, filename):
        self.filename = filename
        self.dset = None
        self.locations = None
        self.node_names = None
        self.interpolated = False
        self.node_loc = {}
        self.main_ran = False
        self.track_names = None
        
        
        # load data when class is initialized
        self._load_data()
        
    def _load_data(self):
        with h5py.File(self.filename, "r") as f:
            dset_names = list(f.keys())
            locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]
            
            self.dset = dset_names
            self.locations = locations
            self.node_names = node_names
            self.track_names = f["track_names"][:]
            self.track_names = [str(i).split("'")[1] for i in list(f["track_names"][:])]

        
        
    def get_info(self):
        
        print("===filename===")
        print(self.filename)
        print()

        print("===HDF5 datasets===")
        print(self.dset)
        print()

        print("===locations data shape===")
        print(self.locations.shape)
        print()

        nodes = {}
        print("===nodes===")
        for i, name in enumerate(node_names):
            nodes[name] = i
        print()
        print(nodes)      
        
        
    def interpolate(self, kind="linear"):
        if self.dset is None:
            print('No video loaded, use load_video() first')
            return
        
        if self.interpolated is True:
            print ('Already filled')
            return
        # store initial shape
        Y = self.locations
        initial_shape = Y.shape
        
        # Flatten after first dim.
        Y = Y.reshape((initial_shape[0], -1)) 
        
        # Interpolate along each slice
        for i in range(Y.shape[-1]):
            y = Y[:, i]

            # Build interpolant.
            x = np.flatnonzero(~np.isnan(y))
            f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

            # Fill missing
            xq = np.flatnonzero(np.isnan(y))
            y[xq] = f(xq)

            # Fill leading or trailing NaNs with the nearest non-NaN values
            mask = np.isnan(y)
            y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

            # Save slice
            Y[:, i] = y

        # Restore to initial shape.
        Y = Y.reshape(initial_shape)
        
        self.locations = Y
        self.interpolated = True
        
        return
    
    def get_node_locations(self):
        # if video not yet loaded:
        if self.dset is None:
            print('No video loaded, use load_video() first')
            return
        
        # not yet interpolated
        if not self.interpolated:
            print('Data not interpolated yet, please use interpolate()')
            return
            
        for i in self.node_names:
            curr_node = f'{i}'
            self.node_loc[curr_node] = self.locations[:, self.node_names.index(i), :, : ]
    
    
    def visualize_movement(self, node="nose", save_to_svg=False):
        """
        Visualize the movement of a specific node (default: 'nose').
        """
        #Set track colors for by and cap mice
        TRACK_COLORS = {
            "by": "#143b42",  # Teal color for "by"
            "cap": "#ec644f"  # Poppy color for cap
        }

        if not self.node_loc:
            print("Node locations not yet initialized, please use get_node_locations().")
            return

        node_loc = self.node_loc[node]

        sns.set("notebook", "ticks", font_scale=1.2)
        mpl.rcParams["figure.figsize"] = [15, 15]

        lw = 2  # Line width

        # Plot spatial trajectory
        plt.figure(figsize=(7, 7))
        for track_name in TRACK_COLORS.keys():  # Iterate over track names in the color mapping
            if track_name in self.track_names:
                i = self.track_names.index(track_name)  # Get the index of the track name
                color = TRACK_COLORS[track_name]  # Get the corresponding color
                plt.plot(
                    node_loc[:, 0, i], node_loc[:, 1, i],
                    color=color,
                    label=track_name,
                    lw=lw,
                    alpha=0.8
                )

        plt.xticks([])
        plt.yticks([])
        plt.legend(loc="upper right")
        plt.title(f"{node} Spatial Trajectory")

        if save_to_svg:
            # Create a "trajectories" folder in the same directory as the HDF5 file
            output_dir = os.path.join(os.path.dirname(self.filename), "trajectories")
            os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

            # Save the SVG file in the "trajectories" folder
            output_path = os.path.join(output_dir, f"{os.path.basename(self.filename).replace('.h5', '')}_{node}_movement2.svg")
            plt.savefig(output_path, format="svg")
            print(f"Saved SVG to: {output_path}")
    
    def main(self, node = 'nose', save = False, plot = False):
        if self.main_ran:
            print("Already")
        self._load_data()
        self.interpolate()
        self.get_node_locations()
        self.main_ran = True
        
        if plot:
            self.visualize_movement(node = node, save_to_svg = save)

        return
    
    
    
    
        
        

