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

def sample_run(pck, tneural, rot = 'roll'):
    run = np.array([pck[k][rot] for k in range(len(pck)-1)])
    run[0] = 0
    
    trun = np.array([pck[k]['nSamples'] for k in range(len(pck)-1)])
    trun = trun - np.diff(trun).mean()/2
    
    if len(tneural)!=len(trun):
        crun = np.cumsum(run)
        crun_neuro = np.interp(tneural, trun, crun)
        run_neuro = np.diff(crun_neuro).astype('float32')
        run_neuro = np.concatenate((run_neuro, run_neuro[:1]), 0)
    else:
        run_neuro = run
    return run_neuro, run, trun

def prediction_wrapper(input_data, Vsv, tneural, spks, U, tcam=None):
    if input_data.shape[0] != Vsv.shape[0]:
        input_data = input_data.T
    x = input_data
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    # declare the model
    pred_model = nn_prediction.PredictionNetwork(n_in=x.shape[-1], n_kp=30, 
                                                n_out=Vsv.shape[-1])
    pred_model.to(device);
    # PC prediction
    if tcam is None:
        y_pred_all, ve_all, itest = pred_model.train_model(x, Vsv, np.arange(tneural.size), np.arange(tneural.size), 
                                                           delay=-1, learning_rate=1e-3, n_iter=400,
                                                            device=device, verbose=True)
    else:
        y_pred_all, ve_all, itest = pred_model.train_model(x, Vsv, tcam, tneural, delay=-1,
                                                            learning_rate=1e-3, n_iter=400,
                                                            device=device, verbose=True)
    Vpred_nl = y_pred_all.reshape(-1, Vsv.shape[-1])
    itest = itest.flatten() # we run using batches of data so we now flatten it
    varexp_pcs = 1 - (((Vpred_nl - Vsv[itest])**2).mean(axis=0)) / ((Vsv[itest]**2).mean(axis=0))
    varexp_pcs_mean = 1 - (((Vpred_nl - Vsv[itest])**2).mean(axis=0)).mean() / ((Vsv[itest]**2).mean(axis=0)).mean()
    Vpred_nl = U @ y_pred_all.reshape(-1, Vsv.shape[-1]).T #U @ y_pred_all.T
    varexp_neurons = 1 - ((Vpred_nl - spks.T[itest].T)**2).mean(axis=1) / (spks.T[itest].T**2).mean(axis=1)
    varexp_neurons_mean = varexp_neurons.mean()
    return varexp_pcs, varexp_pcs_mean, varexp_neurons, varexp_neurons_mean, itest, Vpred_nl

def get_sampling_rate(tstamps):
    """Calculate the sampling rate given timestamps in seconds."""
    min_time = np.min(tstamps)
    time = tstamps.copy()
    time -= min_time
    time *= 24 * 60 * 60  # Convert to seconds
    sampling_rate = 1 / np.diff(time).mean()
    return sampling_rate

def behav_prediction_wrapper(input_data, Vsv, tneural):
    if input_data.shape[0] != Vsv.shape[0]:
        input_data = input_data.T
    x = input_data
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    # declare the model
    pred_model = nn_prediction.PredictionNetwork(n_in=x.shape[-1], n_kp=30, 
                                                n_out=Vsv.shape[-1])
    pred_model.to(device);
    # PC prediction
    y_pred_all, ve_all, itest = pred_model.train_model(x, Vsv, np.arange(tneural.size), np.arange(tneural.size), delay=-1,
                                                            learning_rate=1e-3, n_iter=400,
                                                            device=device, verbose=True)
    return ve_all

# Resample pupil to neural timestamps
def load_pupil(root, db):
    """ Teturn: Timeline """
    p = os.path.join(root,db['mname'],db['datexp'],db['blk'])
    f = db['mname']+'_'+db['datexp']+'_'+db['blk']+'_pupil.npy'
    if os.path.exists(os.path.join(p,f)):
        pupil = np.load(os.path.join(p,f))
    else:
        pupil = []
    return pupil

def Get_eye(root,db):
    faceInfo={}
    facInd, facTim, fac_yxs= [], [], []
    eyeInfo = load_pupil(root,db)
    if not isinstance(eyeInfo,list):
        facInd, facTim, fac_yxs = np.arange(0,eyeInfo.shape[0],1),Get_facedn(eyeInfo[:,:7]),eyeInfo[:,7:]
    else:
        behRes = load_beh(root, db)['BehaviorResults']
        if 'Pup_Msg_data' in behRes.keys():
            if behRes['Pup_Msg_data'].shape[0]>0:
                behRes['facInd'] = behRes['Pup_Msg_data'][:0]
                behRes['facTim'] = Get_facedn(behRes['Pup_Msg_data'][:,4:])
                behRes['fac_yxs'] = behRes['Pup_Msg_data'][:,1:4] 
    faceInfo={'facInd':facInd,'facTim':facTim,'fac_yxs':fac_yxs}
    return faceInfo

def Get_facedn(dt):
    dn = np.empty((dt.shape[0],1))
    for i in range(dt.shape[0]):
#         dn[i] = datetime2matlabdn(dt[i,:].astype(int))
        dn[i] = datetime2datenum(dt[i,:].astype(int))
    return dn

def datetime2datenum(dt):
    """ dt: list or array, y:m:d:h:m:s:ms """
    import datetime
    dtime = datetime.datetime(dt[0], dt[1], dt[2], dt[3], dt[4], dt[5], dt[6])
    mdn = dtime + datetime.timedelta(days = 366)
    frac_seconds = (dtime-datetime.datetime(dtime.year,dtime.month,dtime.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dtime.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds

def load_beh(root,db):
    """ Teturn: Timeline """
    p = os.path.join(root,db['mname'],db['datexp'],db['blk'])
    f = 'Timeline_'+db['mname']+'_'+db['datexp']+'_'+db['blk']+'.npy'
    mat_fn = os.path.join(p,'Timeline_'+db['mname']+'_'+db['datexp']+'_'+db['blk']+'.mat')
    rawmat_fn = os.path.join(p,'Timeline_'+db['mname']+'_'+db['datexp']+'_'+db['blk']+'_RAW.mat')
    if os.path.isfile(os.path.join(p,f)):
        print('loading behaviral .npy file...')
        Timeline = np.load(os.path.join(p,f),allow_pickle = True).item()
    elif os.path.isfile(os.path.join(p,mat_fn)):
#         import Preprocess
        print('loading behaviral .mat file...')
#         Timeline = Preprocess.GetBehavior(root, db)
        Timeline = load_beh_mat(mat_fn, rawmat_fn)
    else:
        print(os.path.join(p,f)+' is not found....')
        print(os.path.join(p,mat_fn)+' is not found....')
    Timeline['ExpInfo'] = db['mname']+'_'+db['datexp']+'_'+db['blk']
    Timeline['db'] = db
    print('Done loading.')
    return Timeline

def load_beh_mat(mat_fn=[],rawmat_fn='input_file_name_here'):
    """return Timeline"""
    def loadh5py(filename): # read Timeline from .mat file
        import h5py
        Timeline = h5py.File(filename, 'r')['Timeline']
        return Timeline
    if not any(mat_fn):
        raise Exception('file name is wrong')
    data = np.array(loadh5py(rawmat_fn)['data']).T if os.path.isfile(rawmat_fn) else [] # load data from -7.3 mat

    Timeline_temp = loadmat(mat_fn)['Timeline']
    BehaviorResults_temp = Timeline_temp[0,0]['BehaviorResults']
    BehaviorSettings_temp = Timeline_temp[0,0]['BehaviorSettings'] 

    BehaviorResults, BehaviorSettings, Settings, Timeline = {}, {}, {}, {}
    names = BehaviorResults_temp.dtype.names
    for i in names:
        BehaviorResults[i] = BehaviorResults_temp[0,0][i]
    names = BehaviorSettings_temp.dtype.names
    for i in names:
        BehaviorSettings[i] = BehaviorSettings_temp[0,0][i]
    if 'Settings' in Timeline_temp.dtype.names:
        Settings_temp = Timeline_temp[0,0]['Settings']    
        names = Settings_temp.dtype.names
        for i in names:
            Settings[i] = Settings_temp[0,0][i]
        
    Timeline['BehaviorResults'] = BehaviorResults
    Timeline['BehaviorSettings'] = BehaviorSettings
    Timeline['Settings'] = Settings     
    Timeline['data'] = data
    return Timeline

def get2Pft(anal_sig=[], t=[],return_all=False):
   """anal_sig: analog signal in data
      t: time for each batch in data
   """ 
   if len(anal_sig)>1000:
      ind = np.where(t>0)[0]
      chunkTime = t[ind]
      frame_start_temp = (anal_sig[:-1] > 2.5) & (anal_sig[1:] <= 2.5)
      frame_start = np.where(frame_start_temp==1)[0]
      ft = np.interp(frame_start, ind, chunkTime) 
   else:
      ft = []
   if return_all:
      chunkInd = ind
      return ft, chunkTime, chunkInd
   else:
      return ft
   
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

# Following code taken from: https://github.com/esdalmaijer/PyGazeAnalyser/tree/master
# Dalmaijer, E.S., Mathôt, S., & Van der Stigchel, S. (2013). PyGaze: an open-source, cross-platform toolbox for minimal-effort programming of eye tracking experiments. Behaviour Research Methods. doi:10.3758/s13428-013-0422-2

def remove_missing(x, y, time, missing):
	mx = np.array(x==missing, dtype=int)
	my = np.array(y==missing, dtype=int)
	x = x[(mx+my) != 2]
	y = y[(mx+my) != 2]
	time = time[(mx+my) != 2]
	return x, y, time

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


def detect_saccades(x, y, time, speed_threshold_percentile=95, minlen=0):

    """Detects saccades, defined as consecutive samples with an inter-sample
	velocity of over a velocity threshold or an acceleration threshold
	
	arguments

	x		-	numpy array of x positions
	y		-	numpy array of y positions
	time		-	numpy array of tracker timestamps in milliseconds
    speed_threshold_percentile - percentile of speed to use as threshold
	
	returns
	Ssac, Esac
			Ssac	-	list of lists, each containing [starttime]
			Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
	"""
    
    Ssac = []
    Esac = []

    # Inter-sample distance
    intdist = (np.diff(x)**2 + np.diff(y)**2)**0.5
    # Inter-sample times
    inttime = np.diff(time)
    # Speed 
    speed = intdist / inttime

    speed_threshold = np.nanpercentile(speed, speed_threshold_percentile)

    # Get saccade start and end times
    t0i = 0
    stop = False
    while not stop:
        # saccade start (t1) is when the velocity or acceleration
        # surpass threshold, saccade end (t2) is when both return
        # under threshold
    
        # detect saccade starts
        sacstarts = np.where((speed[1+t0i:] > speed_threshold).astype(int) >= 1)[0]
        if len(sacstarts) > 0:
            # timestamp for starting position
            t1i = t0i + sacstarts[0] + 1
            if t1i >= len(time)-1:
                t1i = len(time)-2
            t1 = time[t1i]
            
            # add to saccade starts
            Ssac.append([t1])
            
            # detect saccade endings
            sacends = np.where((speed[1+t1i:] < speed_threshold).astype(int) == 1)[0]
            if len(sacends) > 0:
                # timestamp for ending position
                t2i = sacends[0] + 1 + t1i + 2
                if t2i >= len(time):
                    t2i = len(time)-1
                t2 = time[t2i]
                dur = t2 - t1

                # ignore saccades that did not last long enough
                if dur >= minlen:
                    # add to saccade ends
                    Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i], t1i, t2i])
                else:
                    # remove last saccade start on too low duration
                    Ssac.pop(-1)

                # update t0i
                t0i = 0 + t2i
            else:
                stop = True
        else:
            stop = True
            
    return Ssac, Esac


def saccade_detection(x, y, time, inttime=None, missing=0.0, minlen=0, maxvel=100000, maxacc=100000):
	
    """Detects saccades, defined as consecutive samples with an inter-sample
    velocity of over a velocity threshold or an acceleration threshold

    arguments

    x		-	numpy array of x positions
    y		-	numpy array of y positions
    time		-	numpy array of tracker timestamps in milliseconds

    keyword arguments

    missing	-	value to be used for missing data (default = 0.0)
    minlen	-	minimal length of saccades in milliseconds; all detected
                saccades with len(sac) < minlen will be ignored
                (default = 5)
    maxvel	-	velocity threshold in pixels/second (default = 40)
    maxacc	-	acceleration threshold in pixels / second**2
                (default = 340)

    returns
    Ssac, Esac
            Ssac	-	list of lists, each containing [starttime]
            Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
    """
    #x, y, time = remove_missing(x, y, time, missing)

    # CONTAINERS
    Ssac = []
    Esac = []

    # INTER-SAMPLE MEASURES
    # the distance between samples is the square root of the sum
    # of the squared horizontal and vertical interdistances
    intdist = (np.diff(x)**2 + np.diff(y)**2)**0.5
    # get inter-sample times
    if inttime is None:
        inttime = np.diff(time)
    # recalculate inter-sample times to seconds
    #inttime = inttime #/ 1000.0
	
    # VELOCITY AND ACCELERATION
    # the velocity between samples is the inter-sample distance
    # divided by the inter-sample time
    vel = intdist / inttime
    # the acceleration is the sample-to-sample difference in
    # eye movement velocity
    acc = np.diff(vel)

    # SACCADE START AND END
    t0i = 0
    stop = False
    while not stop:
        # saccade start (t1) is when the velocity or acceleration
        # surpass threshold, saccade end (t2) is when both return
        # under threshold
	
        # detect saccade starts
        sacstarts = np.where((vel[1+t0i:] > maxvel).astype(int) + (acc[t0i:] > maxacc).astype(int) >= 1)[0]
        if len(sacstarts) > 0:
            # timestamp for starting position
            t1i = t0i + sacstarts[0] + 1
            if t1i >= len(time)-1:
                t1i = len(time)-2
            t1 = time[t1i]
            
            # add to saccade starts
            Ssac.append([t1])
			
            # detect saccade endings
            sacends = np.where((vel[1+t1i:] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
            if len(sacends) > 0:
                # timestamp for ending position
                t2i = sacends[0] + 1 + t1i + 2
                if t2i >= len(time):
                    t2i = len(time)-1
                t2 = time[t2i]
                dur = t2 - t1

                # ignore saccades that did not last long enough
                if dur >= minlen:
                    # add to saccade ends
                    Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
                else:
                    # remove last saccade start on too low duration
                    Ssac.pop(-1)

                # update t0i
                t0i = 0 + t2i
            else:
                stop = True
        else:
            stop = True

    return Ssac, Esac

def butter_lowpass_filter(data, cutoff_freq, sampling_rate, order=5):
    # Nyquist frequency
    nyquist = 0.5 * sampling_rate
    # Normalized cutoff frequency (as a fraction of Nyquist frequency)
    normal_cutoff = cutoff_freq / nyquist
    
    # Design the Butterworth filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the data
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

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
    
def compute_psth_vectorized(spikes, saccade_idx, window=(-10, 10), sigma=10):
    """
    Compute PSTH aligned to saccade indices.
    
    Parameters:
        spikes (ndarray): shape (n_neurons, n_timepoints), non-integer values allowed.
        saccade_idx (array-like): Indices of saccades in time.
        window (tuple): Time window (before, after) saccade in ms.
        sigma (int): Standard deviation for Gaussian smoothing.
        
    Returns:
        psth (ndarray): Mean spike rate (Hz) over aligned trials.
        bin_centers (ndarray): Time bin centers relative to saccade.
    """
    num_neurons, num_timepoints = spikes.shape
    window = np.array(window)
    bin_edges = np.arange(window[0], window[1] + 1)
    
    # Initialize PSTH array
    saccade_psth = np.zeros((num_neurons, len(bin_edges) - 1))
    
    # Compute PSTH
    valid_saccades = 0
    for idx in saccade_idx:
        if idx + window[0] >= 0 and idx + window[1] < num_timepoints:
            aligned_spikes = spikes[:, idx + window[0]:idx + window[1]]
            saccade_psth += aligned_spikes
            valid_saccades += 1
    
    # Normalize by the number of valid saccades
    psth = saccade_psth / valid_saccades
    
    # Smooth the PSTH
    if sigma > 0:
        psth = gaussian_filter1d(psth, sigma=sigma, axis=1)
    
    return psth, bin_edges

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

# median filter pupil positions
def median_filter(data, window_size):
    """
    Apply median filter to the data
    """
    filtered_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        start_idx = max(0, i - window_size)
        end_idx = min(data.shape[0], i + window_size)
        filtered_data[i] = np.median(data[start_idx:end_idx])
    return filtered_data

# Function to find nearest derivative peaks in original traces
def find_exact_saccade_indices(saccade_regions, x, y, time, threshold):
    exact_saccades = []

    intdist = (np.diff(x)**2 + np.diff(y)**2)**0.5
    inttime = np.diff(time)
    speed = intdist / inttime
    acc = np.diff(speed) # acceleration
    # Calculate the speed threshold
    speed_threshold = np.nanpercentile(speed, threshold)
    print(speed.shape, acc.shape, x.shape)

    for start_filt, end_filt in saccade_regions:
        # Consider a search window around detected saccades
        search_start = max(0, start_filt - 5)
        search_end = min(len(x), end_filt + 10)
        # Find the first large deviation in the derivative (saccade onset)        
        start_exact = search_start + np.where(speed[search_start:search_end] > speed_threshold)[0]
        # if start_exact is not found, skip this saccade
        if len(start_exact) == 0:
            continue
        start_exact = start_exact[0]
        end_exact = np.where(speed[1+start_exact:search_end] < speed_threshold)[0]
        if len(end_exact) == 0:
            end_exact = start_exact + 1
        else:
            end_exact = end_exact[0] + start_exact + 1
        if len(exact_saccades)>0 and start_exact in np.array(exact_saccades)[:,0]: # remove saccade repititions
            continue

        # Store speed and acceleration values for the exact saccade
        exact_saccades.append((start_exact, end_exact, time[start_exact], time[end_exact], 
                               speed[start_exact-1], speed[end_exact-1], acc[start_exact-2], acc[end_exact-2],
                                x[start_exact], y[start_exact], x[end_exact], y[end_exact]))

    return exact_saccades

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

    """
    # Load the video
    cap = cv2.VideoCapture('your_video.mp4', cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video")
    else:
        print("Video opened successfully")
    # Get the properties of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read the first frame as the reference
    frames = []

    # Read the rest of the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # convert the frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    # Convert the frames to a numpy array
    frames = np.array(frames)
    # Close the video
    cap.release()
    """
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

def baseline_signal(data, win_baseline=5, sig_baseline=10):
    """
    win_baseline : float
        window (in seconds) for max filter

    sig_baseline : float
        width of Gaussian filter in frames
    """
    data_baselined = gaussian_filter(data, [0, sig_baseline])
    data_baselined = minimum_filter1d(data_baselined, win_baseline)
    data_baselined = maximum_filter1d(data_baselined, win_baseline)
    data_baselined = data - data_baselined
    return data_baselined

def get_dF_F(F, Fneu, baseline_window=60, baseline_sigma=10):
    """
    Compute dF/F for a given signal
    """
    F_corrected = F - 0.7*Fneu
    # Compute baseline signal
    baseline = baseline_signal(F_corrected, win_baseline=baseline_window, sig_baseline=baseline_sigma)
    # Compute dF/F
    #dF_F = (F_corrected - baseline) / baseline
    return F_corrected

def get_neural_pc(neural_activity, n_comps=128):
    neural_activity -= neural_activity.mean(axis=1)[:,np.newaxis]
    std = ((neural_activity**2).mean(axis=1)**0.5)[:,np.newaxis]
    std = np.where(std==0, 1, std)  # don't scale when std==0
    neural_activity /= std
    #neural_activity /= ((neural_activity**2).mean(axis=1)**0.5)[:,np.newaxis]
    model = PCA(n_components=n_comps, copy=False).fit(neural_activity)
    # obtain neural PCs (project PCs onto raw data): time x components
    Y = model.components_.T * model.singular_values_
    return Y
