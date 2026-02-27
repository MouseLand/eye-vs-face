from fig3 import get_light_random_info, get_light_static_info, load_data
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys, os
from scipy.stats import zscore
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
from rastermap import Rastermap, utils
import matplotlib.gridspec as gridspec
from itertools import groupby


LABEL_FONTSIZE = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = LABEL_FONTSIZE
plt.rcParams['axes.titlesize'] = LABEL_FONTSIZE
plt.rcParams['axes.labelsize'] = LABEL_FONTSIZE
plt.rcParams['xtick.labelsize'] = LABEL_FONTSIZE-1
plt.rcParams['ytick.labelsize'] = LABEL_FONTSIZE-1
plt.rcParams['legend.fontsize'] = LABEL_FONTSIZE
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.titlelocation"] = "left"
plt.rcParams["axes.titleweight"] = "normal"

pupil_label_color = 'green'
face_eye_label_color = 'orange'
face_eye_pupil_label_color = 'red'
pupil_x_color = 'darkgreen'
pupil_y_color = 'lightgreen'

# Set folder
BASE_FOLDER = 'set_your_base_folder_here/' 

def set_path(data_path):
    global BASE_FOLDER
    BASE_FOLDER = data_path

def plot(data_path, recording_type, mname):
    
    set_path(data_path)
    raster_plot_light_predictions(recording_type, mname)

def get_behavioral_data(recording_type):

    if recording_type == 'static':
        dbs = get_light_static_info() 
        mname = 'TX140'
    else:
        dbs = get_light_random_info() 
        mname = 'TX144' 

    datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]

    # Check if file exists 
    if os.path.exists(f"{BASE_FOLDER}data/mouse_data/{mname}/{datexp}/{mname}_{datexp}_rasterdata.npz"):
        data = np.load(f"{BASE_FOLDER}data/mouse_data/{mname}/{datexp}/{mname}_{datexp}_rasterdata.npz", allow_pickle=True)
        X_embedding = data['X_embedding']
        aligned_saccades = data['aligned_saccades']
        pupil_x = data['pupil_x']
        pupil_y = data['pupil_y']
        exact_saccades = data['exact_saccades']
        pupil_timestamp = data['pupil_timestamp']
        spks_timestamp = data['spks_timestamp']
        motSVD = data['motSVD']

    else:
        sys.path.append(f"{BASE_FOLDER}analysis/code/")
        import helper_functions
        _, _, _, _, _, _, _, npz_data = load_data(f"{BASE_FOLDER}data/mouse_data/{mname}/{datexp}/", mname, datexp)
        spks = []
        for block_idx in range(1): # CHANGE THIS IDX
            block = list(npz_data['tcam'].item().keys())[block_idx]
            # Resample behavior data to neural timestamps
            spks.append(helper_funtions.segment_data(npz_data['spks'].T, npz_data['neural_frames_list'], block_idx).T)
        spks = np.concatenate(spks, axis=1)

        pupil_x = npz_data['pupil_x_small']
        pupil_y = npz_data['pupil_y_small']
        blink = npz_data['blink_small']
        
        blocks = list(npz_data['tcam'].item())
        blk = blocks[0].split('blk')[1]
        blink_baselined_threshold = float([entry['blink_baselined_threshold'] for entry in dbs if entry['mname'] == mname and entry['blk']==blk][0])

        # Baseline correction for blink artifacts
        blink_baselined = gaussian_filter(blink, 10)
        blink_baselined = minimum_filter1d(blink_baselined, 5)
        blink_baselined = maximum_filter1d(blink_baselined, 5)
        blink_baselined = blink - blink_baselined
        pupil_x_filtered = helper_functions.interpolate_pupil_blink(pupil_x, blink_baselined, blink_baselined_threshold)
        pupil_y_filtered = helper_functions.interpolate_pupil_blink(pupil_y, blink_baselined, blink_baselined_threshold)
        pupil_timestamp = npz_data['tcam'].item()['blk1']
        spks_timestamp = npz_data['tneural'].item()['blk1']

        # Filter out high-frequency noise in pupil data
        cutoff_freq = 1.5 # Adjust this value depending on the movement frequency you want to filter out (in Hz)
        tmin = np.min(pupil_timestamp)
        cam_time = (pupil_timestamp - tmin) * 24 * 60 * 60
        sampling_rate = (1 / np.mean(np.diff(cam_time))).astype(int)
        pupil_x_filtered = helper_functions.butter_lowpass_filter(pupil_x_filtered, cutoff_freq, sampling_rate)
        pupil_y_filtered = helper_functions.butter_lowpass_filter(pupil_y_filtered, cutoff_freq, sampling_rate)

        speed_threshold_percentile = float([entry['speed_threshold'] for entry in dbs if entry['mname'] == mname and entry['blk']==blk][0]) 
        Ssac, Esac = helper_functions.detect_saccades(pupil_x_filtered, pupil_y_filtered, pupil_timestamp, speed_threshold_percentile)
        Sblk, Eblk = helper_functions.detect_blinks(blink_baselined, pupil_timestamp, blink_baselined_threshold)
        # Filter any saccades that are between blinks
        Esac, Eblk = np.array(Esac), np.array(Eblk)
        start_saccades = Esac[:, -2]
        end_saccades = Esac[:, -2]
        start_blinks = Eblk[:, -2]
        end_blinks = Eblk[:, -2]
        filtered_saccades = []
        for i in range(len(Esac)):
            if not any((start_saccades[i] >= start_blinks-10) & (start_saccades[i] <= end_blinks+10)): ### Change this to 0.5 seconds before and after; and get derivative of blink to get blink_start
                filtered_saccades.append(Esac[i])
        filtered_saccades = np.array(filtered_saccades)

        saccade_regions = []
        for sac in filtered_saccades:
            start, end = sac[-2].astype(int), sac[-1].astype(int)
            saccade_regions.append((start, end))
        exact_saccades = helper_functions.find_exact_saccade_indices(saccade_regions, pupil_x, pupil_y, pupil_timestamp, threshold=speed_threshold_percentile)

        model = Rastermap().fit(zscore(spks.copy(), axis=1))
        y = model.embedding # neurons x 1
        isort = model.isort
        X_embedding = zscore(utils.bin1d(spks[isort], bin_size=200, axis=0), axis=1)
        
        # Save data to npz file 
        motSVD = npz_data['face_svd'].item()['motSVD'][0]

    return X_embedding, pupil_x, pupil_y, exact_saccades, pupil_timestamp, spks_timestamp, motSVD

def remove_spines(ax):
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)


def raster_plot_light_predictions(recording_type, mname):

    if recording_type == 'static':
        dbs = get_light_static_info() 
    else:
        dbs = get_light_random_info() 


    # Load rastermap output from nonlinearmodel predictions
    rastermap_out = np.load(f"{BASE_FOLDER}data/figure_data/light{recording_type}_rastermap_out_udcnv.npy", allow_pickle=True).item()

    datexp = [db['datexp'] for db in dbs if db['mname'] == mname][0]
    _, _, _, _, _, _, _, npz_data = load_data(f"{BASE_FOLDER}data/mouse_data/{mname}/{datexp}/", mname, datexp)
   
    # Get isort if saved 
    data = np.load(f"{BASE_FOLDER}data/rasterdata/{mname}_{datexp}_rasterdata_udcnv.npz", allow_pickle=True)
    isort = data['isort']

    # Load neural and pupil data
    xlim_neural = [0, 1500]
    n_folds = 5
    vline_width = .4

    # Plot rastermap of before and after predictionss
    y_test_all, y_pred_face_all, y_pred_pupil_all = [], [], []
    condition = 'before'
    tstart = rastermap_out[mname]['pupil']['tneural_start']
    tend = rastermap_out[mname]['pupil']['tneural_end']

    # Get raw data
    y_test_all = np.empty((rastermap_out[mname]['pupil']['U'].shape[0], 0))
    for k_idx in range(n_folds):
        for segment in rastermap_out[mname]['pupil'][condition]['y_test'][k_idx]:
            if segment.shape[0] > 0:
                y_test = (segment @ rastermap_out[mname]['pupil']['U'].T)
                y_test_all = np.hstack((y_test_all, y_test.T))

    # Get neural data from all k-folds and concatenate them: Pupil
    y_pred_pupil_all = np.empty((rastermap_out[mname]['pupil']['U'].shape[0], 0))
    for k_idx in range(n_folds):
        for segment in rastermap_out[mname]['pupil']['before']['y_pred'][k_idx]:
            if segment.shape[0] > 0:
                y_test = (segment @ rastermap_out[mname]['pupil']['U'].T)
                y_pred_pupil_all = np.hstack((y_pred_pupil_all, y_test.T))

    # Get prediction from face w/o eye
    y_pred_face_all = np.empty((rastermap_out[mname]['face_eye']['U'].shape[0], 0))
    for k_idx in range(n_folds):
        for segment in rastermap_out[mname]['face_eye'][condition]['y_pred'][k_idx]:
            if segment.shape[0] > 0:
                y_test = (segment @ rastermap_out[mname]['face_eye']['U'].T)
                y_pred_face_all = np.hstack((y_pred_face_all, y_test.T))

    # Get test indices from each k_fold and combine them into a single list
    data_splits_itest = rastermap_out[mname]['pupil']['data_splits_ineural_test']
    itest_blocks = []
    for fold in data_splits_itest:        # loop over k folds
        for seg in fold[0]:       # loop over segments in each fold
            for element in seg:
                itest_blocks.append(element)   
    itest_sorting = np.argsort(itest_blocks)

    x_test = []
    for k_idx in range(n_folds):
        x_test_pupil = rastermap_out[mname]['pupil'][condition]['x_test'][k_idx]
        itest_sample_b = rastermap_out[mname]['pupil'][condition]['itest_sample_b'][k_idx]
        for s_idx, segment in enumerate(x_test_pupil):
            val = segment[itest_sample_b[s_idx]]
            x_test.append(val)
    x_test_pupil = np.concatenate(x_test, axis=0)

    x_test = []
    for k_idx in range(n_folds):
        x_test_motsvd = rastermap_out[mname]['face_eye'][condition]['x_test'][k_idx]
        itest_sample_b = rastermap_out[mname]['face_eye'][condition]['itest_sample_b'][k_idx]
        for s_idx, segment in enumerate(x_test_motsvd):
            val = segment[itest_sample_b[s_idx]]
            x_test.append(val)
    x_test_motsvd = np.concatenate(x_test, axis=0)

    print(x_test_pupil.shape, y_test_all.shape, y_pred_pupil_all.shape)

    saccade_data = np.load(f'{BASE_FOLDER}data/saccade_data/saccade_times.npy', allow_pickle=True).item()
    aligned_saccades = []
    # Get saccades
    for i, block in enumerate(saccade_data["light_{}".format(recording_type)][mname].keys()):
        _, a_sac = get_aligned_saccade_times(saccade_data["light_{}".format(recording_type)][mname][block], npz_data['tneural'].item()['blk{}'.format(block)])
        a_sac = a_sac.astype(int)
        if i==0:
            aligned_saccades.append(a_sac)
    aligned_saccades = np.hstack(aligned_saccades)
    mask = (aligned_saccades >= tstart) & (aligned_saccades <= tend)
    saccades_new = aligned_saccades[mask] - tstart

    # z-score the predicted data using mean and std of test data
    y_test_mean = np.nanmean(y_test_all, axis=1, keepdims=True)
    y_test_std = np.nanstd(y_test_all, axis=1, keepdims=True)
    y_test_all = (y_test_all - y_test_mean) / y_test_std
    y_pred_face_all = (y_pred_face_all - y_test_mean) / y_test_std
    y_pred_pupil_all = (y_pred_pupil_all - y_test_mean) / y_test_std

    # Sort the data based on the model's sorting and bin it
    y_test_embedding = utils.bin1d(y_test_all[isort], bin_size=200, axis=0)
    y_pred_face_embedding = utils.bin1d(y_pred_face_all[isort], bin_size=200, axis=0)
    y_pred_pupil_embedding = utils.bin1d(y_pred_pupil_all[isort], bin_size=200, axis=0)

    print("y_test_embedding shape:", y_test_embedding.shape)

    # Colormap with transparent NaNs
    cmap = plt.cm.gray_r.copy()
    cmap.set_bad(color=(0, 0, 0, 0))

    fig = plt.figure(figsize=(8, 7), dpi=300)

    # First subplot (Raw)
    ax1 = plt.subplot(511)
    raster = ax1.imshow(y_test_embedding[:,itest_sorting], aspect='auto', vmin=0, vmax=1, cmap=cmap)
    ax1.set_title('Test data')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(xlim_neural)
    remove_spines(ax1)  
    # Add y-axis scale bar for rastermap
    bin_size = 200
    scale_bar_length = 25
    scale_bar_start = ax1.get_ylim()[0] - 50
    scale_bar_end = scale_bar_start + scale_bar_length
    # orient the text vertically
    ax1.plot([ax1.get_xlim()[0]-(0.01*np.diff(ax1.get_xlim())), ax1.get_xlim()[0]-(0.01*np.diff(ax1.get_xlim()))],
              [scale_bar_start, scale_bar_end], 'k-', lw=2, clip_on=False)
    ax1.text(ax1.get_xlim()[0]-(0.03*np.diff(ax1.get_xlim())), scale_bar_start-50, f'{scale_bar_length*bin_size} neurons',
              ha='left', va='center', rotation='vertical', fontsize=LABEL_FONTSIZE-3)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0], pos[1], pos[2], pos[3]*1.3])
    # Add x-axis scale bar for pupil area, x and y, rastermap on the right
    duration = 10 # seconds
    scale_bar_length = 3 * duration
    scale_bar_start = ax1.get_xlim()[1] - 70
    scale_bar_end = scale_bar_start + scale_bar_length
    y_pos = ax1.get_ylim()[1]-5
    ax1.plot([scale_bar_start, scale_bar_end], [y_pos, y_pos], 'k-', lw=1, clip_on=False)
    ax1.text(scale_bar_start + scale_bar_length/2, y_pos, f'{int(duration)} s'.format(str(duration)), 
             ha='center', va='bottom', clip_on=False, fontsize=LABEL_FONTSIZE-2)
    
    # Get the current position of the axis (ax) in figure coordinates
    pos = ax1.get_position()  # [x0, y0, width, height]

    # Manually define the position for the colorbar
    cax = fig.add_axes([
        pos.x0 + pos.width + 0.003,  # x-coordinate (a bit to the right of the main plot)
        pos.y0+.1,                     # y-coordinate (aligned with the main plot)
        pos.width * 0.007,           # width of the colorbar (relative to plot width)
        pos.height *.35                 # height of the colorbar (same as the main plot)
    ])
    # Create the colorbar
    plt.colorbar(raster, cax=cax, orientation="vertical")
    # Add a label to the colorbar
    cax.set_ylabel("z-scored", labelpad=-.9, fontsize=LABEL_FONTSIZE-3)  
    cax.set_yticks([0, 1], ['0', '1'], fontsize=LABEL_FONTSIZE-4)
    
    # Second subplot (Predicted)
    ax2 = plt.subplot(512)
    ax2.imshow(y_pred_face_embedding[:,itest_sorting], aspect='auto', vmin=0, vmax=1, cmap=cmap)
    ax2.set_title('Face w/o eye', color='darkorange')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim(xlim_neural)
    remove_spines(ax2)  
    pos = ax2.get_position().bounds
    ax2.set_position([pos[0], pos[1]-.056, pos[2], pos[3]*1.3])

    # Third subplot (Pupil)
    ax3 = plt.subplot(513)
    ax3.imshow(y_pred_pupil_embedding[:,itest_sorting], aspect='auto', vmin=0, vmax=1, cmap=cmap)
    ax3.set_title('Pupil position', color='darkgreen')
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_xlim(xlim_neural)
    remove_spines(ax3)
    pos = ax3.get_position().bounds
    ax3.set_position([pos[0], pos[1]-.115, pos[2], pos[3]*1.3])
    
    # Fourth subplot (Pupil X and Y)
    ax4 = plt.subplot(514)
    ax4.plot(x_test_pupil[:,0][itest_sorting], label='Azimuth', lw=.5, color=pupil_x_color)
    ax4.plot(x_test_pupil[:,1][itest_sorting], label='Elevation', lw=.5, color=pupil_y_color)
    ax4.set_xlim(xlim_neural)
    ax4.set_xticks([])
    ax4.margins(x=0)   
    remove_spines(ax4)  
    ax4.set_yticks([])
    legend=ax4.legend(loc='center left', bbox_to_anchor=(-0.15, 0.47),facecolor='none', fontsize=LABEL_FONTSIZE-2)
    legend.get_texts()[0].set_color(pupil_x_color)  
    legend.get_texts()[1].set_color(pupil_y_color)  
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)
    # Add y-axis scale bar for pupil x and y
    scale_bar_length = 10
    scale_bar_start = ax4.get_ylim()[0] + 25
    scale_bar_end = scale_bar_start + scale_bar_length
    pos = ax4.get_position().bounds
    ax4.set_position([pos[0], pos[1]-.01, pos[2], pos[3]*.3])
    
    # Fourth subplot 
    ax5 = plt.subplot(515)
    line, = ax5.plot(x_test_motsvd[:,0][itest_sorting], label='Motion\nSVD', color='orange', lw=1.2)
    legend=ax5.legend(loc='center left', bbox_to_anchor=(-0.15, 0.55), facecolor='none', fontsize=LABEL_FONTSIZE-2)
    legend.get_texts()[0].set_color('orange') 
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)    
    ax5.set_xlim(xlim_neural)
    ax5.margins(x=0)
    remove_spines(ax5)
    ax5.set_yticks([])
    ax5.set_xticks([])
    # adjust pos of ax5 
    pos = ax5.get_position().bounds
    ax5.set_position([pos[0], pos[1]+.085, pos[2], pos[3]*.3])
    
    # add saccade lines to all plots
    for s in saccades_new:
        ax1.axvline(s, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
        ax2.axvline(s, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
        ax3.axvline(s, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
        ax4.axvline(s, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
        ax5.axvline(s, color=(0,1,0), linestyle='dashed', linewidth=vline_width)

    plt.savefig(f"{BASE_FOLDER}figures/supp_rastermap_{recording_type}_{mname}_{datexp}.png", bbox_inches='tight', dpi=300)

def get_orig_idx(plotted_idx, chunk_len, test_idx_ranges):
    k = plotted_idx // chunk_len
    offset = plotted_idx % chunk_len
    orig_start = test_idx_ranges[k][0]
    orig_idx = orig_start + offset
    return orig_idx

def get_aligned_saccade_times(saccade_data, spks_timestamps):
    """
    Aligns saccade times to neural timestamps.
    
    Parameters:
    - saccade_data: list of 4 values: saccade_start_idx, saccade_end_idx, saccade_start_t, saccade_end_t
    - spks_timestamps: ndarray of neural timestamps 
    
    Returns:
    - neural_saccade_times: ndarray of shape (N) containing the nearest neural timestamps for each saccade
    """
    saccade_start_t = np.array(saccade_data)[:,2].astype(float)
    # Find nearest neural timestamps to saccade start times
    neural_saccade_times = np.zeros(len(saccade_start_t))
    neural_saccade_idx = np.zeros(len(saccade_start_t))
    for i, t in enumerate(saccade_start_t):
        try:
            start_time = np.argmin(np.abs(spks_timestamps - t))
            neural_saccade_times[i] = spks_timestamps[start_time]
            neural_saccade_idx[i] = start_time
        except:
            continue
    return neural_saccade_times, neural_saccade_idx

def in_any_range(x, ranges):
    return any(start <= x <= end for start, end in ranges)

def ranges_from_indices(indices):
    indices = sorted(map(int, indices))  # ensure sorted + cast to int
    ranges = []
    for _, g in groupby(enumerate(indices), lambda x: x[1] - x[0]):
        g = list(g)
        ranges.append((int(g[0][1]), int(g[-1][1])))
    return ranges
