import rpy2
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
numpy2ri.activate()
pandas2ri.activate()

from skimage.draw import circle
from skimage.restoration import denoise_wavelet
from skimage.feature import peak_local_max

from scipy.ndimage.filters import gaussian_laplace
from scipy.stats import gumbel_r
from scipy import spatial

import multiprocessing

import numpy as np
import pandas as pd

import pims
import trackpy as tp
from trackpy.preprocessing import bandpass

from math import sqrt
import math

from functools import partial

import xml.etree.ElementTree as ET

from itertools import product
from os.path import basename

# Generate necessary R functions first
# R function for fitting a gumbel GLM to the local maxima in blob detection
# Estimates the gumbel parameters as a function of sigma (for blob) and local background
rpy2.robjects.r("source('puff_lib.R')")

gev_glm = rpy2.robjects.globalenv['gev_glm']
get_pc_scores = rpy2.robjects.globalenv['get_pc_scores']

# The blob_log function calls the helper function _prune_blobs to remove overlapping detected blobs
# This function is not accessible from the module, so I've just pasted the source code here
# We want to have access to _prune_blobs so that we can remove overlapping blobs when we combine the
# detected blobs across different thresholds - when we use a different threshold for each sigma, we're doing
# a call to blob_log for each sigma, but we then need to remove any overlaps between different sigmas.

def _compute_disk_overlap(d, r1, r2):
    """
    Compute fraction of surface overlap between two disks of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.
    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first disk.
    r2 : float
        Radius of the second disk.
    Returns
    -------
    fraction: float
        Fraction of area of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -
            0.5 * sqrt(abs(a * b * c * d)))
    return area / (math.pi * (min(r1, r2) ** 2))


def _compute_sphere_overlap(d, r1, r2):
    """
    Compute volume overlap fraction between two spheres of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.
    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first sphere.
    r2 : float
        Radius of the second sphere.
    Returns
    -------
    fraction: float
        Fraction of volume of the overlap between the two spheres.
    Notes
    -----
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
    """
    vol = (math.pi / (12 * d) * (r1 + r2 - d)**2 *
           (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2))
    return vol / (4./3 * math.pi * min(r1, r2) ** 3)


def _blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs.
    Returns a float representing fraction of overlapped area.
    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    n_dim = len(blob1) - 1
    root_ndim = sqrt(n_dim)

    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim

    d = sqrt(np.sum((blob1[:-1] - blob2[:-1])**2))
    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1

    if n_dim == 2:
        return _compute_disk_overlap(d, r1, r2)

    else:  # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return _compute_sphere_overlap(d, r1, r2)


def _prune_blobs(blobs_array, overlap=0.5):
    """Eliminated blobs with area overlap.
    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 (or 4) values,
        ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
        where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
        and ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.
        This array must not have a dimension of size 0.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if _blob_overlap(blob1, blob2) > overlap:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.array([b for b in blobs_array if b[-1] > 0])

# This is a helper function for parallelizing find_locs
def _find_locs_in_frame(idx_frame, sigma_list, cutoff):
    """Single frame processor for movies. Denoises, blob detects,
    and calculates background information to identify potential events
    Parameters
    ----------
    idx_frame : list
        A 2-element list, containing the frame number (int) and frame (pims.Frame) to be analysed
    sigma_list : list
        A list of ints indicating at sigmas the Laplace-on-Gaussian should be evaluated
    cutoff: int
        The quantile at which the fitted distribution of background noise should be thresholded to select
        specific signal
    Returns
    -------
    bls : pandas.DataFrame
        DataFrame containing frame number and the y and x coordinates of each blob detected in the frame.
    """
    # Denoise the frame, apply laplacian for each standard deviation, and find
    # the local max values of the laplacian
    idx = idx_frame[0]
    frame = idx_frame[1]
    #frame = denoise_wavelet(frame, multichannel=False)
    frame = bandpass(frame, 1, 15, 1)
    gls = [-gaussian_laplace(frame, sig) * sig **2 for sig in sigma_list]
    plm = [peak_local_max(x) for x in gls]
    plmval = np.concatenate([[gls[i][r, c] for (r, c) in plm[i]] for i in range(len(sigma_list))])
    sigmas_of_peaks = np.concatenate([np.repeat(sigma_list[i], len(plm[i])) for i in range(len(sigma_list))])
    plm = np.hstack([np.concatenate(plm), sigmas_of_peaks.reshape(len(sigmas_of_peaks), 1)])

    loc_background = np.zeros(len(plm))
    for i, loc_max in enumerate(plm):
        rr, cc = circle(loc_max[0], loc_max[1], 9)
        cc_new = cc[np.where((0 <= rr) & (rr <= frame.shape[0] - 1) & (0 <= cc) & (cc <= frame.shape[1] - 1))]
        rr_new = rr[np.where((0 <= rr) & (rr <= frame.shape[0] - 1) & (0 <= cc) & (cc <= frame.shape[1] - 1))]
        loc_background[i] = np.median(frame[rr_new, cc_new])

    coef = gev_glm(plmval, sigmas_of_peaks, loc_background)
    thresh = gumbel_r.ppf(q=cutoff,
                          loc=coef[0] + coef[1]*sigmas_of_peaks + coef[2]*loc_background + coef[3]*sigmas_of_peaks*loc_background,
                          scale=coef[4] +coef[5]*sigmas_of_peaks + coef[6]*loc_background + coef[7]*sigmas_of_peaks*loc_background)
    plm = plm[np.where(plmval > thresh)]
    bls = _prune_blobs(plm)

    # record current frame number, rather than the sigma used in blob detection
    bls[:,2] = idx

    # Important note! blob_log function returns (row, col, sigma)
    # row corresponds to y and column to x
    bls = pd.DataFrame(bls, columns=['y', 'x', 'frame'])
    return bls[['frame', 'x', 'y']]

# Function to detect blobs in cell videos
# Returns a pandas data frame with columns for (x,y) location and frame number (0-indexed) for detected blobs
# The output of this function can then be the input for particle tracking to link blobs across frames

# f: array or list of frames from cell video
# max_sigma: largest standard deviation considered for LoG blob detection
# min_sigma: smallest standard deviation considered for LoG blob detection
# num_sigma: number of standard deviations to try in blob detection
# cutoff: the cutoff used to find an intensity threshold

def find_locs(f, max_sigma=3, min_sigma=1, num_sigma=11, cutoff=0.9):
    
    # Generate list of evenly-spaced standard deviations between min_sigma and max_sigma
    scale = np.linspace(0, 1, num_sigma)[:, None]
    sigma_list = scale * (max_sigma - min_sigma) + min_sigma
    sigma_list = sigma_list[:,0]
    
    n_cores = multiprocessing.cpu_count()
    f_with_sigmas = partial(_find_locs_in_frame, sigma_list=sigma_list, cutoff=cutoff)
    with multiprocessing.Pool(3) as pool:
        blobs_out = pool.map(f_with_sigmas, enumerate(f), chunksize=100)
    
    # To do particle tracking across frames, after calling this function you would run the following:
    #
    # events = tp.link_df(locs, search_range=search_range, memory=memory)
    # events = tp.filter_stubs(events, track_length_min)
    #
    # for specified values of
    # search_range: restriction on number of pixels the particle can move from frame to frame
    # memory: number of frames the particle can disappear for
    # track_length_min: minimum number of frames a track must exist for 
    
    locs = pd.concat(blobs_out, ignore_index=True)
    
    return locs


# import x coord, y coord, and frame number for all of the puffs identified in an xml file
# makes LOTS of assumptions about how the data are structured
# fix this later!
def import_xml_data(f):
    tree = ET.parse(f)
    root = tree.getroot()
    markers = root[1]
    marker_coords = []
    for m in markers[1]:
        if m.tag == 'Marker':
            marker_coords = marker_coords + [[int(m[0].text), int(m[1].text), int(m[2].text)]]
    
    return marker_coords


# for each event in the xml file (in order), either return the id of the event in the imported matlab file
# or return -1 if we can't find it in the matlab data
# df should be a pandas data frame
# loc is a triple of x coord, y coord, frame number
def filter_df(df, loc, radius=5):
    # match frame, and match (x,y) coords within radius
    id_list = df[(np.abs(df['frame'] - (loc[2] - 1)) < 1) &  (np.abs(df['x'] - loc[0]) < radius) & 
       (np.abs(df['y'] - loc[1]) < radius)]['particle'].tolist()
    if not id_list:
        return - 1
    return id_list[0]


# for each event in the movie, add a few frames to the start and end to make sure we're 
# capturing the full lifetime of the event
# TODO: Parallelize
def pad_frames(puff_events, puff_ids, f, num_pad=5):
    for idx in np.sort(puff_ids[puff_ids > -1]):
        start_x = puff_events['x'][puff_events['particle'] == idx].tolist()[0]
        end_x = puff_events['x'][puff_events['particle'] == idx].tolist()[-1]
        start_y = puff_events['y'][puff_events['particle'] == idx].tolist()[0]
        end_y = puff_events['y'][puff_events['particle'] == idx].tolist()[-1]
        start_frame = puff_events['frame'][puff_events['particle'] == idx].tolist()[0]
        end_frame = puff_events['frame'][puff_events['particle'] == idx].tolist()[-1]
        
        tmp_events = []
    
        for f_num in range(start_frame - num_pad, start_frame):
            tmp_events.append([f_num, start_x, start_y, idx])
        
        for f_num in range(end_frame + 1, end_frame + 1 + num_pad):
            tmp_events.append([f_num, end_x, end_y, idx])
    
    tmp_events = pd.DataFrame(tmp_events, columns=['frame', 'x', 'y', 'particle'])
    tmp_events = tmp_events.sort_values(by=['particle', 'frame'])
    puff_events = puff_events.append(tmp_events, sort=True)
    return puff_events[['frame', 'x', 'y', 'particle']]

# for each event in the movie, fetch the grid of intensity values around the center of the event
# the grid is of dimension (2*delta + 1)x(2*delta + 1)
# TODO: Parallelize
def intensity_grid(f, puff_events, delta=4):
    puff_intensities = []
    delta = 4
    side = 2*delta + 1
    for f_num, xloc, yloc, idx in np.array(puff_events):
        y_len, x_len = np.shape(f[f_num])
        xloc = int(xloc)
        yloc = int(yloc)
        
        # literal edge case detection
        y_start = (yloc - delta) if (yloc - delta) >= 0 else 0
        y_end = (yloc + delta + 1) if (yloc + delta) <= y_len else (y_len + 1)
        x_start = (xloc - delta) if (xloc - delta) >= 0 else 0
        x_end = (xloc + delta + 1) if (xloc + delta) <= x_len else (x_len + 1)
        block = f[f_num][y_start:y_end, x_start:x_end]
        
        if x_start == 0:
            block = np.pad(block, ((0,0), ((delta-xloc)+1,0)), mode="reflect")
        elif x_end == (x_len + 1):
            block = np.pad(block, ((0,0), (0,delta-(x_len-xloc)+1)), mode="reflect")
            
        if y_start == 0:
            block = np.pad(block, (((delta-yloc)+1,0), (0,0)), mode="reflect")
        elif y_end == (y_len + 1):
            block = np.pad(block, ((0,delta-(y_len-yloc)+1), (0,0)), mode="reflect")
        
        for r,c in product(range(side), repeat=2):
            puff_intensities.append([f_num, c - delta, r - delta, idx, block[r,c]])
        
    puff_intensities = pd.DataFrame(puff_intensities, columns=['frame', 'x', 'y', 'particle', 'intensity'])
    return puff_intensities
    

# the full procedure for processing a movie, from blob detection and particle tracking,
# through finding intensity grids for each event, and matching puffs to scored data
# provided in an XML file
# This function returns and saves information on all puffs specified in markers file,
# and also selects a random subset of 100 nonpuffs to analyze in the same way (for
# comparison to puffs)
# movie: the name of the multi-page tiff file containing the cell movie
# markers: the name of an XML file containing info for events classified as puffs
# num_pad: number of frames to add to beginning and end of each event
# delta: determines grid size when extracting grid of intensities around center of detected event
def process_movie(movie, markers, num_pad=5, delta=4, save=True):
    f = pims.TiffStack(movie)
    marker_locs = import_xml_data(markers)
    locs = find_locs(f, cutoff=0.9, max_sigma=3)
    events = tp.link_df(locs, search_range=3, memory=0)
    events = tp.filter_stubs(events, 4)
    if save:
        events.to_csv(basename(movie) + '_events.csv')
    
    puff_ids = np.array([filter_df(events, m, 5) for m in marker_locs])
    puff_events = events[events['particle'].isin(puff_ids)]
    puff_events = pad_frames(puff_events, puff_ids, f, num_pad)
    puff_intensities = intensity_grid(f, puff_events, delta)
    
    nonpuff_ids = np.setdiff1d(np.unique(events['particle']), puff_ids)
    nonpuff_sample = np.random.choice(nonpuff_ids, 100, replace=False)
    nonpuff_events = events[events['particle'].isin(nonpuff_sample)]
    nonpuff_events = pad_frames(nonpuff_events, nonpuff_sample, f, num_pad)
    nonpuff_intensities = intensity_grid(f, nonpuff_events, delta)
    
    if save:
        puff_intensities.to_csv(basename(movie) + '_puff_intensities.csv')
        puff_events.to_csv(basename(movie) + '_puff_events.csv')
        nonpuff_intensities.to_csv(basename(movie) + '_nonpuff_intensities.csv')
        nonpuff_events.to_csv(basename(movie) + '_nonpuff_events.csv')
    return puff_events, puff_intensities, nonpuff_events, nonpuff_intensities