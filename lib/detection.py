import rpy2
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
numpy2ri.activate()
pandas2ri.activate()

import importlib.resources as pkg_resources 

from skimage.draw import circle
from skimage.restoration import denoise_wavelet
from skimage.feature import peak_local_max

from scipy.stats import gumbel_r
from scipy import spatial
from scipy import ndimage
from scipy import stats
from scipy import signal

import multiprocessing

import numpy as np
import pandas as pd

import pims
import trackpy as tp

from math import sqrt
import math

from functools import partial

import xml.etree.ElementTree as ET

from itertools import product
from os.path import basename

from tqdm.auto import tqdm

from lib import analysis

import time

tp.quiet(True)

# Generate necessary R functions first
# R function for fitting a gumbel GLM to the local maxima in blob detection
# Estimates the gumbel parameters as a function of sigma (for blob) and local background
with pkg_resources.path('lib', 'detection.R') as filepath:
    rpy2.robjects.r("source('" + str(filepath) + "')")

gev_glm = rpy2.robjects.globalenv['gev_glm']

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

def get_mask(movie, mode_ratio = 0.8, connect = True):
    """ Gets cell mask from image by thresholding at the minimum after
    the first mode of fluorescence in a maximum intensity projection of the movie
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
    projection = np.amax(movie, axis=0)
    dim_x, dim_y = np.shape(projection)

    blurred = ndimage.filters.gaussian_filter(projection, 5)
    linear_blurred = blurred.flatten()
    percentiles = np.percentile(linear_blurred, [1,99])
    linear_blurred = np.delete(linear_blurred, [(linear_blurred < percentiles[0]) |
                                                (linear_blurred < percentiles[1])])
    intensity_kde = stats.gaussian_kde(linear_blurred)
    intensity_bins, bin_size = np.linspace(linear_blurred.min(), linear_blurred.max(), 100, retstep=True)
    intensity_probs = intensity_kde(intensity_bins)
    local_max = signal.argrelmax(intensity_probs)[0]
    local_min = signal.argrelmin(intensity_probs)[0]

    if local_min is not None:
        first_mode_min_id = np.argmax(local_min>local_max[0])

        if ((first_mode_min_id is not None) and 
            (np.sum(intensity_probs[0:local_min[first_mode_min_id]])*bin_size < mode_ratio)):

            mask = blurred>intensity_bins[local_min[first_mode_min_id]]

            if connect:
                labeled_img, labels = ndimage.label(mask)
                biggest_obj = np.argmax(np.bincount(labeled_img[labeled_img>0].flatten()))
                mask = labeled_img == biggest_obj

        else:
            mask = np.ones((dim_x, dim_y))

    else:
        mask = np.ones((dim_x, dim_y))

    return mask

def get_loc_background(frame, sigma = 1.26, alpha = 0.05):
    frame = frame.astype('float64')
    window = math.ceil(4*sigma)
    x = np.arange(-window, window+1)
    g = np.exp(-x**2/(2.*sigma**2.))
    u = np.ones((1,len(x)))

    extended_frame = np.pad(frame, window, mode='reflect')
    fg = signal.convolve2d(g*g.reshape(-1,1), extended_frame, mode='valid')
    fu = signal.convolve2d(u.reshape(-1,1)*u, extended_frame, mode='valid');
    fu2 = signal.convolve2d(u.reshape(-1,1)*u, extended_frame**2, mode='valid');

    #2-D kernel
    g2 = g*g.reshape(-1,1)
    n = len(g2.flatten());
    gsum = np.sum(g2.flatten());
    g2sum = np.sum(g2.flatten()**2);

    # solution to linear system
    A_est = (fg - gsum*fu/n) / (g2sum - gsum**2/n)
    c_est = (fu - A_est*gsum)/n

    # filter signic regions
    J = np.hstack([g2.flatten().reshape(-1,1), np.ones((n,1))])
    C = np.linalg.inv(np.matmul(J.T,J))

    f_c = fu2 - 2*c_est*fu + n*c_est**2
    RSS = A_est**2*g2sum - 2*A_est*(fg - c_est*gsum) + f_c
    RSS[RSS<0] = 0
    sigma_e2 = RSS/(n-3)

    sigma_A = np.sqrt(sigma_e2*C[0,0])
    
    # residuals for fitting gaussian of given sigma
    sigma_res = np.sqrt(RSS/(n-1))

    kLevel = stats.norm.ppf(1-alpha/2.0)

    SE_sigma_c = sigma_res/np.sqrt(2*(n-1)) * kLevel
    df2 = (n-1) * (sigma_A**2 + SE_sigma_c**2)**2 / (sigma_A**4 + SE_sigma_c**4)
    scomb = np.sqrt((sigma_A**2 + SE_sigma_c**2)/n)
    T = (A_est - sigma_res*kLevel) / scomb
    pval = stats.t.cdf(-T, df2)
    mask = pval < alpha
    snr = A_est/c_est
    return(c_est, mask, sigma_res, snr)

# This is a helper function for parallelizing find_locs
def _find_locs_in_frame(idx_frame, sigma_list, cutoff, 
                        mask=None, filter_points=True,
                        old_coefs = None):
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
    background_frame, signif_mask, residuals, snr = get_loc_background(frame)
    frame = denoise_wavelet(frame, multichannel=False)
    gls = [-ndimage.filters.gaussian_laplace(frame, sig) * sig **2 for sig in sigma_list]
    
    if mask is None:
        mask = np.ones(np.shape(frame))
    if filter_points:
        mask = mask * signif_mask
        
    plm = [peak_local_max(x, indices=False) for x in gls]
    plm = [np.transpose(np.nonzero(x & mask))[::-1] for x in plm]
    plmval = np.concatenate([[gls[i][r, c] for (r, c) in plm[i]] for i in range(len(sigma_list))])
    sigmas_of_peaks = np.concatenate([np.repeat(sigma_list[i], len(plm[i])) for i in range(len(sigma_list))])
    plm = np.hstack([np.concatenate(plm), sigmas_of_peaks.reshape(len(sigmas_of_peaks), 1)])

    loc_background = np.array([background_frame[int(loc_max[0]), int(loc_max[1])] for loc_max in plm])
    try:
        coef = gev_glm(plmval, sigmas_of_peaks, loc_background)
    except rpy2.rinterface.RRuntimeError:
        if old_coefs is None:
            raise ValueError()
        else:
            coef = old_coefs
            
    thresh = gumbel_r.ppf(q=cutoff,
                          loc=coef[0] + coef[1]*sigmas_of_peaks + coef[2]*loc_background + coef[3]*sigmas_of_peaks*loc_background,
                          scale=coef[4] +coef[5]*sigmas_of_peaks + coef[6]*loc_background + coef[7]*sigmas_of_peaks*loc_background)
    plm = plm[np.where(plmval > thresh)]
    if np.shape(plm)[0] == 0:
        return pd.DataFrame([], columns=['frame', 'x', 'y', 'residuals', 'snr']), coef
    else:
        bls = _prune_blobs(plm)

        # record current frame number, rather than the sigma used in blob detection
        bls[:,2] = idx

        # Important note! blob_log function returns (row, col, sigma)
        # row corresponds to y and column to x
        bls = pd.DataFrame(bls, columns=['y', 'x', 'frame'], dtype="int16")
        bls['residuals'] = [residuals[y,x] for y,x in bls[['y','x']].values]
        bls['snr'] = [snr[y,x] for y,x in bls[['y','x']].values]
        return bls[['frame', 'x', 'y', 'residuals', 'snr']], coef

# Function to detect blobs in cell videos
# Returns a pandas data frame with columns for (x,y) location and frame number (0-indexed) for detected blobs
# The output of this function can then be the input for particle tracking to link blobs across frames

# f: array or list of frames from cell video
# max_sigma: largest standard deviation considered for LoG blob detection
# min_sigma: smallest standard deviation considered for LoG blob detection
# num_sigma: number of standard deviations to try in blob detection
# cutoff: the cutoff used to find an intensity threshold

def find_locs(f, max_sigma=3, min_sigma=1, num_sigma=11, cutoff=0.9, mask=True):

    # Generate list of evenly-spaced standard deviations between min_sigma and max_sigma
    scale = np.linspace(0, 1, num_sigma)[:, None]
    sigma_list = scale * (max_sigma - min_sigma) + min_sigma
    sigma_list = sigma_list[:,0]
    
    old_coefs = None
    blobs_out = []
    for idx_frame in enumerate(f):
        if mask:
            f_mask = get_mask(f)
        else:
            f_mask = None
        blobs, old_coefs = _find_locs_in_frame(idx_frame, 
                                               sigma_list=sigma_list, 
                                               cutoff=cutoff, 
                                               mask = f_mask, 
                                               old_coefs=old_coefs)
        blobs_out.append(blobs)

# This is for parallel execution. Doesn't work with old_coefs method
#     n_cores = multiprocessing.cpu_count()

#     if mask:
#         f_mask = get_mask(f)
#         f_with_sigmas = partial(_find_locs_in_frame,
#                                 sigma_list=sigma_list,
#                                 cutoff=cutoff,
#                                 mask=f_mask)
#         with multiprocessing.Pool(3) as pool:
#             blobs_out = pool.map(f_with_sigmas, enumerate(f), chunksize=100)
#     else:
#         f_with_sigmas = partial(_find_locs_in_frame,
#                                 sigma_list=sigma_list,
#                                 cutoff=cutoff)
#         with multiprocessing.Pool(3) as pool:
#             blobs_out = pool.map(f_with_sigmas, enumerate(f), chunksize=100)

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
    id_list = df[(np.abs(df['frame'] - (loc[2] - 1)) < 1) &
                 (np.abs(df['x'] - loc[0]) < radius) &
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

# helper function for parallel intensity_grid
def _intensity_grids_for_frame(frame_and_events, max_frames, delta=4):
    frame = frame_and_events[0]
    puff_events = frame_and_events[1]
    puff_intensities = []
    side = 2*delta + 1
    for f_num, xloc, yloc, idx in np.array(puff_events):
        if (f_num < 0) | (f_num >= max_frames):
            block = np.zeros((9,9))
        else:
            y_len, x_len = np.shape(frame)
            xloc = int(xloc)
            yloc = int(yloc)

            # literal edge case detection
            y_start = (yloc - delta) if (yloc - delta) >= 0 else 0
            y_end = (yloc + delta + 1) if (yloc + delta) <= y_len else (y_len + 1)
            x_start = (xloc - delta) if (xloc - delta) >= 0 else 0
            x_end = (xloc + delta + 1) if (xloc + delta) <= x_len else (x_len + 1)
            block = frame[y_start:y_end, x_start:x_end]

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

# for each event in the movie, fetch the grid of intensity values around the center of the event
# the grid is of dimension (2*delta + 1)x(2*delta + 1)
def intensity_grid(f, puff_events, delta=4):
    frames = np.unique(puff_events['frame'])
    max_frames = np.shape(f)[0]

    frame_generator = (f[n] if (n >= 0
                                and n < max_frames)
                       else 0 for n in frames)
    frame_events_generator = (puff_events[puff_events['frame'] == n]
                              for n in frames)

    n_cores = multiprocessing.cpu_count()
    f_with_options = partial(_intensity_grids_for_frame,
                             max_frames = max_frames,
                             delta=delta)
    with multiprocessing.Pool(3) as pool:
        grids_out = pool.map(f_with_options,
                             zip(frame_generator, frame_events_generator),
                             chunksize=100)

    puff_intensities = pd.concat(grids_out, ignore_index=True)
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
def process_movie(movie, markers=None, 
                  num_pad=5, delta=4, 
                  save=True):
    movie_name = basename(movie)
    
    f = pims.TiffStack(movie)
    average_f = ndimage.uniform_filter(f[:], size=(5,0,0))
    if markers is not None:
        marker_locs = import_xml_data(markers)
    
    t_start = time.time()
    print("Getting events for %s... " % movie_name, end='')
    locs = find_locs(average_f, cutoff=0.9)
    events = tp.link_df(locs, search_range=1.5, memory=0)
    events = tp.filter_stubs(events, 4)
    if markers is not None:
        puff_ids = np.array([filter_df(events, m, 5) for m in marker_locs])
        events['puff'] = events['particle'].isin(puff_ids).astype(int)
    if save:
        events.to_csv(basename(movie) + '_events.csv', index=False)
    t_events = time.time() - t_start
    print("Finished (%d seconds)" % t_events)
    
    print("Getting intensities for %s... " % movie_name, end='')
    intensities = intensity_grid(f,events[['frame','x','y','particle']])
    if markers is not None:
        intensities['puff'] = intensities['particle'].isin(puff_ids).astype(int)
    if save:
        intensities.to_csv(basename(movie) + '_intensities.csv')
    t_intens = (time.time() - t_start) - t_events
    print("Finished (%d seconds)" % t_intens)
    
    print("Getting features for %s... " % movie_name, end='')
    features = analysis.get_features(events, intensities)
    if save:
        features.to_csv(basename(movie) + '_features.csv')
        
    t_features = (time.time() - t_start) - t_intens
    print("Finished (%d seconds)" % t_features)
    
    return events, intensities, features
