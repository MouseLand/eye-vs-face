import os
import numpy as np
from scipy.io import loadmat
import scipy.interpolate as interpolate
from sklearn.decomposition import PCA
import torch
import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
from scipy.interpolate import interp1d
from neuropop import nn_prediction
import pupil_dark_helpers as pdh
import nonlinear_modelv3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def get_neurons_metadata():
    dark_dbs = pdh.carbachol_in_dark() 
    light_random_dbs = pdh.get_light_random_info()
    light_static_dbs = pdh.get_light_static_info()
    base_folders = [
        '/home/asyeda/pupil_analysis/dark_recordings',
        '/home/asyeda/pupil_analysis/light_recordings_random',
        '/home/asyeda/pupil_analysis/light_recordings_static']
    dbs = [dark_dbs, light_random_dbs, light_static_dbs]
    category = ['dark', 'light_random', 'light_static']

    num_neurons = []
    num_recordings = 0
    for db_idx, db in enumerate(dbs):
        base_folder = base_folders[db_idx]
        # Get mouse list
        mouse_list = np.unique([entry['mname'] for entry in db])
        for mname in mouse_list:
            if mname == 'TX150':  # skip this mouse due to different sampling rate
                continue
            print('>>>>>>Processing mouse: ', mname)
            # Get datexp 
            datexp = [entry['datexp'] for entry in db if entry['mname'] == mname][0]
            # Load data
            if mname in ['FXa10', 'FXa8', 'FXa9', 'FXa1']:  # gcamp8 recordings
                spks, _, _, _, _, _, _, _ = nonlinear_modelv3.load_data(f'/home/asyeda/nas113/Atika/{mname}/{datexp}/{1}', mname, datexp) # different base folder for these mice
            else:
                spks_udcnv_file = "/media/Atika/neural_data_udcnv/{}_{}_1_spks.npz".format(mname, datexp)
                spks = np.load(spks_udcnv_file)['spks']

            num_neurons.append(spks.shape[0])
            num_recordings += 1
    # report mean and std number of neurons across recording
    mean_neurons = np.mean(num_neurons)
    std_neurons = np.std(num_neurons)
    
    return mean_neurons, std_neurons
            
def segment_data(data, frames_per_folder, idx):
    idx = np.arange(sum(frames_per_folder[:idx]), sum(frames_per_folder[:idx+1])) 
    return data[idx]

def remove_spines(ax):
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
   
def interp_value(v, vind, tind, fill_value='extrapolate'):
    """Interpolates a value 'v' at a target index 'tind' based on the given value indices 'vind'.

    Args:
        v (list or array-like): The values to be interpolated.
        vind (list or array-like): The indices corresponding to the values.
        tind (float or int): The target index at which to interpolate the value.

    Returns:
        float or int: The interpolated value at the target index.
    """
    Model_ = interpolate.interp1d(vind, v, fill_value=fill_value, bounds_error=False)
    return Model_(tind)


def remove_missing(x, y, time, missing):
	mx = np.array(x==missing, dtype=int)
	my = np.array(y==missing, dtype=int)
	x = x[(mx+my) != 2]
	y = y[(mx+my) != 2]
	time = time[(mx+my) != 2]
	return x, y, time

# Following code taken from: https://github.com/esdalmaijer/PyGazeAnalyser/tree/master
# Dalmaijer, E.S., Mathôt, S., & Van der Stigchel, S. (2013). PyGaze: an open-source, cross-platform toolbox for minimal-effort programming of eye tracking experiments. Behaviour Research Methods. doi:10.3758/s13428-013-0422-2

def detect_blinks(blink_area, time, threshold_percentile, minlen=0):
    # empty list to contain data
    Sblk = []
    Eblk = []
    threshold = np.nanpercentile(blink_area, threshold_percentile)

    # Get saccade start and end times
    t0i = 0
    stop = False
    while not stop:
        # Blink starts when area falls below threshold and ends when it rises above
        starts = np.where((blink_area[t0i:] < threshold).astype(int) >= 1)[0]

        if len(starts) > 0:
            # timestamp for starting position
            t1i = starts[0] + t0i
            if t1i >= len(time)-1:
                t1i = len(time)-2
            t1 = time[t1i]

            # add to blink starts
            Sblk.append([t1])

            # detect blink endings
            ends = np.where((blink_area[t1i:] > threshold).astype(int) == 1)[0]
            if len(ends) > 0:
                # timestamp for ending position
                t2i = ends[0] + t1i
                t2 = time[t2i]
                dur = t2 - t1

                # ignore blinks that did not last long enough
                if dur >= minlen:
                    # add to blink ends
                    Eblk.append([t1, t2, dur, blink_area[t1i], blink_area[t2i], t1i, t2i])
                else:
                    # remove last blink start on too low duration
                    Sblk.pop(-1)
            else:
                # remove last blink start if no end was found
                Sblk.pop(-1)

            # update t0i
            t0i = 0 + t2i
        else:
            stop = True
    else:
        stop = True

    return Sblk, Eblk


def interpolate_pupil_blink(pupil: np.ndarray, blink: np.ndarray, percentile: float) -> np.ndarray:
    """
    Interpolates pupil data where blink values are below a specified percentile threshold.

    Parameters:
    -----------
    pupil : np.ndarray
        Array of pupil values.
    blink : np.ndarray
        Array of blink values.
    percentile : float
        Percentile value to threshold blink data (e.g., 5 for 5th percentile).

    Returns:
    --------
    pupil_interp : np.ndarray
        Array of pupil values with interpolated values where blink < threshold.
    """
    # Step 1: Calculate the percentile threshold of blink values
    blink_threshold = np.percentile(blink, percentile)

    # Step 2: Create a mask for blink values below the threshold
    blink_mask = blink < blink_threshold

    # Step 3: Find valid and invalid indices
    valid_indices = np.where(~blink_mask)[0]  # Indices where blink is valid
    invalid_indices = np.where(blink_mask)[0]  # Indices where blink is invalid

    # Step 4: Perform linear interpolation on pupil data
    interp_func = interp1d(valid_indices, pupil[valid_indices], kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    
    # Step 5: Replace pupil values where blink < threshold
    pupil_interp = pupil.copy()
    pupil_interp[blink_mask] = interp_func(invalid_indices)

    return pupil_interp
    
def get_aligned_saccade_times(Esac, spks_timestamps):
    """
    Aligns saccade times to neural timestamps.
    
    Parameters:
    - Esac: list of lists where each entry contains [t1, t2]
    - spks_timestamps: ndarray of neural timestamps (sorted)
    
    Returns:
    - aligned_saccades: ndarray of shape (N, 2) where each row contains [aligned_t1, aligned_t2]
    """
    Esac = np.array(Esac)  # Convert to ndarray for easier slicing
    t1_times = Esac[:, 0]  # Extract saccade start times

    # Find nearest neural timestamps
    neural_saccade_times = np.zeros(len(Esac))
    for i, t in enumerate(t1_times):
        try:
            start_time = np.argmin(np.abs(spks_timestamps - t))
            neural_saccade_times[i] = start_time
        except:
            continue
    return neural_saccade_times


def load_video_frames(video_path):

    import imageio

    reader = imageio.get_reader(video_path, 'ffmpeg')

    frames = []
    for frame in reader:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    frames = np.array(frames)

    # Get FPS (frames per second)
    fps = reader.get_meta_data()['fps']
    reader.close()

    return frames, fps


def update_brightness(image, brightness_factor):
    """
    Update the brightness of the image
    """
    brightness_factor = float(brightness_factor)/100
    # Ensure that the brightness factor is within a valid range
    brightness_factor = max(0.0, min(brightness_factor, 2.0))
    # Adjust the brightness by multiplying with the factor
    brightened_image_array = image * brightness_factor
    # Make sure the values are still in the valid 0-1 range
    brightened_image_array = np.clip(brightened_image_array, 0.0, 1.0)
    return brightened_image_array

def remove_spines(ax):
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)
