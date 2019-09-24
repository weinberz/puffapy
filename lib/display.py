from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Takes dataframe with 'frame', 'x', 'y', and 'intensity' columns
# Returns nummpy matrix with dimensions (frames, x, y)
def get_frames(event):
    event = event.sort_values(['frame','y','x'])
    num_frames = len(np.unique(event['frame']))
    dim_x = len(np.unique(event['x']))
    dim_y = len(np.unique(event['y']))
    frames = event['intensity'].values.reshape((num_frames, dim_x, dim_y))
    return(frames)

# Takes event_frames as returned by get_frame
# Plots as montage of images
def plot_montage(event_frames, cmap='gray', frames_per_row=8):
    num_frames = np.shape(event_frames)[0]
    fig_size = (10,1.75*ceil(num_frames/frames_per_row))
    fig, axs = plt.subplots(nrows = ceil(num_frames/frames_per_row), 
                            ncols = frames_per_row,
                            figsize=fig_size)
    axs = axs.ravel()
    vmax = np.amax(event_frames)
    vmin = np.amin(event_frames)
    for frame in range(len(axs)):
        if frame < num_frames:
            axs[frame].imshow(event_frames[frame, :, :], cmap=cmap, vmax=vmax, vmin=vmin)
        axs[frame].axis('off')
    
    plt.tight_layout()    
    return((fig, axs))