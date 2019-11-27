from math import ceil, floor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import ipywidgets as widgets

from IPython.display import HTML

from skimage.transform import resize


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
def plot_montage(event_frames, cmap='gray',
                 frames_per_row=24, figw=15, figh=0.875):
    
    vmax = np.amax(event_frames)
    vmin = np.amin(event_frames)
    num_frames, dim_y, dim_x = event_frames.shape
    
    rows = ceil(num_frames/frames_per_row)
    figh = figh*rows
    fig_size = (figw,figh)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1,1,1)

    pad_frames = np.full(((frames_per_row*rows)-num_frames, 
                           dim_y, 
                           dim_x),
                        fill_value=vmax,
                        dtype=event_frames.dtype)

    frames_to_display = np.concatenate([event_frames,pad_frames])
    frames_to_display = resize(frames_to_display, (frames_per_row*rows , dim_y*4, dim_x*4), order=0, preserve_range=True)
    frames_to_display = np.pad(frames_to_display, 
                               pad_width=((0,0),(1,1),(1,1)),
                               constant_values=vmax,
                               mode='constant')

    frames_to_display = np.vstack([
        np.hstack([
            frames_to_display[(row*frames_per_row)+col,:,:] for col in range(frames_per_row)
        ]) for row in range(rows)
    ])

    ax.imshow(frames_to_display,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.axis('off')
    return((fig, ax))

def plot_summary(event, cmap='gray',
                 frames_per_row=24, figw=15, figh=0.875):
    frames = get_frames(event)
    vmin = np.amin(frames)
    vmax = np.amax(frames)
    fig = plt.figure(figsize=(5,5))
    ims = [[plt.imshow(resize(frame, (frame.shape[0]*2,frame.shape[1]*2), preserve_range=True), 
                       vmin=vmin, vmax=vmax, animated=True, cmap=cmap)] 
           for frame in frames]

    plt.axis('off')
    anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    plt.close()

    vid = widgets.Output()
    vid.append_display_data(HTML(anim.to_html5_video()))

    int_plot = widgets.Output()
    y, x = np.shape(frames[0])
    middle_x = floor(x/2)
    middle_y = floor(y/2)
    ys = [np.amax(frame[middle_y-1:middle_y+1,
                        middle_x-1:middle_x+1])
          for frame in frames]
    fig = plt.figure(figsize=(6,6))
    plt.plot(np.arange(0, len(frames)), ys)
    with int_plot:
        display(fig)
    plt.close()

    montage_plot = widgets.Output()
    fig, axs = plot_montage(frames, frames_per_row=frames_per_row,
                            figw=figw, figh=figh)
    with montage_plot:
        display(fig)
    plt.close()

    widget_list = (int_plot, vid, montage_plot)
    return widget_list