from fig_utils import *
import matplotlib.pyplot as plt 
import os, sys
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import zscore
import helper_functions
from rastermap import Rastermap, utils
from scipy.stats import f_oneway, ttest_rel
from statsmodels.stats.multitest import multipletests
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
sys.path.insert(0, '/home/asyeda/Facemap/facemap')
from facemap import pupil
import pupil_dark_helpers as pdh
from itertools import combinations
import nonlinear_refactored
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import matplotlib.patheffects as pe
import cv2
from sklearn.decomposition import PCA
from scipy.stats import wilcoxon

LABEL_FONTSIZE = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = LABEL_FONTSIZE
plt.rcParams['axes.titlesize'] = LABEL_FONTSIZE
plt.rcParams['axes.labelsize'] = LABEL_FONTSIZE
plt.rcParams['xtick.labelsize'] = LABEL_FONTSIZE-1
plt.rcParams['ytick.labelsize'] = LABEL_FONTSIZE-1
plt.rcParams['legend.fontsize'] = LABEL_FONTSIZE
pupil_x_color = 'darkgreen'
pupil_y_color = 'lightgreen'

def plot():
    fig = plt.figure(figsize=(14, 7), dpi=300)
    gs = GridSpec(2, 4, figure=fig, hspace=.15, wspace=0.2, width_ratios=[1, 1.2, 1.2, .7])

    plot_illustration(gs[0,0], '/home/asyeda/pupil_analysis/dark_fig_illus.png')
    plot_fve_prepost_carbachol(gs[1,0])
    #raster_plot_dark(gs[0,1:3])
    raster_plot_predictions(fig, gs[:,1:3])

    plot_varexp_lineplot(gs[0,3])
    plot_varexp_curve(gs[1,3])
    #scatter_plot(gs[1,2])

    # save figure
    fig.tight_layout()
    plt.savefig('figure2.png', dpi=300, bbox_inches='tight')

    plt.show()

def plot_illustration(gs, fp):
    before_carbachol_video = "/home/asyeda/pupil_analysis/dark_recordings/LZ14_2024_06_20_1_full.mp4"
    after_carbachol_video = "/home/asyeda/pupil_analysis/dark_recordings/LZ14_2024_06_20_2_full.mp4"
    before_eye_video = "/home/asyeda/pupil_analysis/dark_recordings/LZ14_2024_06_20_1_small.mp4"
    after_eye_video = "/home/asyeda/pupil_analysis/dark_recordings/LZ14_2024_06_20_2_small.mp4"
    after_small_proc = "/home/asyeda/pupil_analysis/dark_recordings/LZ14_2024_06_20_2_small_proc.npy"
    after_full_proc = "/home/asyeda/pupil_analysis/dark_recordings/LZ14_2024_06_20_1_full_proc.npy"
    before_idx, after_idx = 26749, 14000 #68414
    alpha = 3 # Contrast multiplier
    beta = 1    # Brightness offset (can be removed or adjusted)

    nested_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs, hspace=-.15, wspace=0.15)
    ax0 = plt.subplot(nested_gs[0, 0])
    frame = get_frame_by_index(before_carbachol_video, before_idx)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ax0.imshow(cv2.convertScaleAbs(frame, alpha=alpha, beta=beta), cmap='gray')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # Turn off ticks and tick labels
    ax0.set_ylabel("Face view", fontsize=LABEL_FONTSIZE)
    ax0.set_xlim(30, 230)
    ax0.set_ylim(230, 50)  # Invert y-axis to match the original image orientation
    ax0.set_title("Before carbachol", fontsize=LABEL_FONTSIZE-1)

    ax1 = plt.subplot(nested_gs[0, 1])
    frame = get_frame_by_index(after_carbachol_video, after_idx)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ax1.imshow(cv2.convertScaleAbs(frame, alpha=alpha, beta=beta), cmap='gray')
    ax1.axis('off')
    ax1.set_xlim(30, 230)
    ax1.set_ylim(230, 50)  # Invert y-axis to match the original image orientation
    ax1.set_title("After carbachol", fontsize=LABEL_FONTSIZE-1)

    # Set a shared title for the top row
    ax0.text(1, 1.2, "Recording in darkness",
         ha='center', va='bottom',
         fontsize=LABEL_FONTSIZE,
         transform=ax0.transAxes, fontweight='bold')

    overlay_image = plt.imread('/home/asyeda/pupil_analysis/paper_figures/pipette_resized.png' )  
    # resize overlay image
    new_width, new_height = int(overlay_image.shape[1]*.7), int(overlay_image.shape[0]*.7)
    overlay_image = cv2.resize(overlay_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # shift pixels in overlay image along x-axis using np.roll
    overlay_image = np.roll(overlay_image, 20, axis=1)  # Shift right by 20 pixels
    ax1.imshow(overlay_image)  # Adjust extent as needed

    alpha = 5.5  # Contrast multiplier
    beta = 1     # Brightness offset (can be removed or adjusted)

    ax2 = plt.subplot(nested_gs[1, 0])
    frame = get_frame_by_index(before_eye_video, before_idx)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ax2.imshow(cv2.convertScaleAbs(frame, alpha=alpha, beta=beta), cmap='gray')
    # Turn off only the spines and ticks, but keep the axis label
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # Turn off ticks and tick labels
    ax2.set_ylabel("Eye view", fontsize=LABEL_FONTSIZE)

    ax3 = plt.subplot(nested_gs[1, 1])
    frame = get_frame_by_index(after_eye_video, after_idx)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ax3.imshow(cv2.convertScaleAbs(frame, alpha=alpha, beta=beta), cmap='gray')
    
    proc_data = np.load(after_small_proc, allow_pickle=True).item()
    proc_frame_idx = after_idx 

    pupil_sigma = proc_data['rois'][1]['pupil_sigma']
    ellipse = proc_data['rois'][1]['ellipse']
    sat = proc_data['rois'][1]['saturation']
    xrange = proc_data['rois'][1]['xrange']
    yrange = proc_data['rois'][1]['yrange']
    crop_x = [proc_data['rois'][1]['xrange'][0], proc_data['rois'][1]['xrange'][-1]+1]
    crop_y = [proc_data['rois'][1]['yrange'][0], proc_data['rois'][1]['yrange'][-1]+1]
    rois = proc_data['rois']
    # Crop the image
    img = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    if img.ndim > 2:
        img = img.mean(axis=-1)
    # smooth in space
    fr = img.astype(np.float32)
    fr[~ellipse] = 255.0
    fr = 255.0 - fr
    fr = np.maximum(0, fr - (255.0 - sat))
    mu, sig, _, _, xy, immiss = pupil.fit_gaussian(
        fr.copy(), pupil_sigma, do_xy=True
    )
    u, sv = proc_data['pupil'][0]['axdir'][proc_frame_idx], proc_data['pupil'][0]['axlen'][proc_frame_idx]
    n = 100  # Number of points around ellipse
    p = np.linspace(0, 2 * np.pi, n)[:, np.newaxis]
    xy = np.concatenate((np.cos(p), np.sin(p)), axis=1) * (sv**0.5) @ u
    xy += mu
    pupil_center = np.concatenate((mu[np.newaxis, :], xy), axis=0)

    # Plot the pupil center and ellipse
    ax3.scatter(xy[:, 1]+crop_x[0], xy[:, 0]+crop_y[0], c='olive', s=.1, 
               marker='+', zorder=5, edgecolors='none')
    ax3.scatter(mu[1]+crop_x[0], mu[0]+crop_y[0], c='green', s=25, marker='+')
    ax3.axis('off')
    
    # Adjust height and position of the subplots
    ax = [ax0, ax1, ax2, ax3]
    for a in ax:
        pos = a.get_position().bounds
        a.set_position([pos[0]-.01, pos[1]+.05, pos[2]*1.12, pos[3]*1.2])
    
def get_frame_by_index(video_path, frame_idx):
    """
    Retrieves a specific frame from a video file.

    Parameters:
        video_path (str): Path to the video file.
        frame_idx (int): Index of the frame to retrieve (0-based).

    Returns:
        frame (numpy.ndarray or None): The retrieved frame as an image, or None if the frame cannot be read.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return None
    
    # Set the video position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    # Read the frame
    ret, frame = cap.read()
    # Release the video capture object
    cap.release()
    if not ret:
        print(f"Error: Unable to retrieve frame at index {frame_idx}")
        return None
    
    return frame

def scatter_plot(gs):
    
    base_folder = '/home/asyeda/pupil_analysis/dark_recordings'
    data = np.load(f"{base_folder}/nonlinear_dark_results_nonshifted.npy", allow_pickle=True).item()

    ax = plt.subplot(gs)
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.02, pos[1], pos[2], pos[3]])

    num_animals = len(data)
    for i in range(num_animals):
        mname = list(data.keys())[i]

        pupil_ve = data[mname]['pupil']["varexp_neurons_after_carbachol"]*100
        face_ve = data[mname]['face_eye']["varexp_neurons_after_carbachol"]*100
        ax.scatter(pupil_ve, face_ve, s=.4, color='grey', alpha=0.5)
        ax.set_xlabel("Pupil variance explained (%)")
        ax.set_ylabel("Face w/o eye\nvariance explained (%)")
        ax.set_xlim(-10, 70)
        ax.set_ylim(-10, 70)
        ax.plot([-100, 100], [-100, 100], 'k--')

    # Adjust position
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.015, pos[1], pos[2]*.9, pos[3]*.9])

"""
def get_behavioral_data(recording_type):

    dbs = pdh.carbachol_in_dark() 
    base_folder = '/home/asyeda/pupil_analysis/dark_recordings/'
    mname = 'LZ14'
    datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]

    blink_idx = []
    # Get blinks for the first carbachol recording only
    blink_data = np.load('/home/asyeda/pupil_analysis/blink_times.npy', allow_pickle=True).item()
    for blk in blink_data[recording_type][mname].keys():
        blink_start_idx = blink_data[recording_type][mname][blk][:,-2].astype(int)
        blink_end_idx = blink_data[recording_type][mname][blk][:,-1].astype(int)
        for s, e in zip(blink_start_idx, blink_end_idx):
            if e > s:
                blink_idx = np.append(blink_idx, np.arange(s-5, e+6)) # few frames before and after blink removed
            else:
                blink_idx = np.append(blink_idx, s)
    blink_idx = np.unique(blink_idx.astype(int))

    # Check if file exists 
    if os.path.exists(f"{base_folder}/{mname}_{datexp}_rasterdata.npz"):
        data = np.load(f"{base_folder}/{mname}_{datexp}_rasterdata.npz", allow_pickle=True)
        # Load blink data
        X_embedding = data['X_embedding']
        aligned_saccades = data['aligned_saccades']
        pupil_x = data['pupil_x']
        pupil_y = data['pupil_y']
        exact_saccades = data['exact_saccades']
        pupil_timestamp = data['pupil_timestamp']
        spks_timestamp = data['spks_timestamp']
        motSVD = data['motSVD']
        neural_frames_list = data['neural_frames_list']
        cam_frames_list = data['cam_frames_list']
        pupil_area = data['pupil_area']
    else:
        # Load saccade data
        saccade_data = np.load('/home/asyeda/pupil_analysis/saccade_times.npy', allow_pickle=True).item()
        _, _, _, _, _, _, _, npz_data = nonlinear_refactored.load_data(base_folder, mname, datexp)

        # Need data concatenated across blocks
        spks = npz_data['spks']
        pupil_x = npz_data['pupil_x_small']
        pupil_y = npz_data['pupil_y_small']
        pupil_area = npz_data['pupil_area_small']

        # Get saccades for the first carbachol recording only i.e. second block
        blocks = list(npz_data['tcam'].item())
        blk = blocks[1].split('blk')[1]
        exact_saccades = saccade_data[recording_type][mname][blk] # Get exact saccade indices from precomputed data
        
        model = Rastermap().fit(zscore(spks.copy(), axis=1))
        y = model.embedding # neurons x 1
        isort = model.isort
        X_embedding = zscore(utils.bin1d(spks[isort], bin_size=200, axis=0), axis=1)
        
        neural_saccade_times, aligned_saccades = get_aligned_saccade_times(exact_saccades, npz_data['tneural'].item()['blk2'])
        aligned_saccades = aligned_saccades.astype(int)
        
        # Get sampling rate of data
        tmin = np.min(npz_data['tcam'].item()['blk2'])
        cam_time = (npz_data['tcam'].item()['blk2'] - tmin) * 24 * 60 * 60
        sampling_rate = (1 / np.mean(np.diff(cam_time))).astype(int)

        # Save data to npz file 
        motSVD = npz_data['face_svd'].item()['motSVD'][0]
        neural_frames_list = npz_data['neural_frames_list']
        cam_frames_list = npz_data['cam_frames_list']

        data = {"X_embedding": X_embedding, "aligned_saccades": aligned_saccades, "pupil_x": pupil_x, "pupil_y": pupil_y, 
                "pupil_timestamp": npz_data['tcam'], "exact_saccades": exact_saccades, "neural_saccade_times": neural_saccade_times, "pupil_area": pupil_area,
                  "spks_timestamp": npz_data['tneural'], "motSVD": motSVD, "sampling_rate": sampling_rate, "neural_frames_list": npz_data['neural_frames_list'],
                  "cam_frames_list": npz_data['cam_frames_list']}  
        np.savez(f"{base_folder}/{mname}_{datexp}_rasterdata.npz", **data)

    return X_embedding, aligned_saccades, pupil_x, pupil_y, exact_saccades, pupil_timestamp, spks_timestamp, motSVD, blink_idx, neural_frames_list, cam_frames_list, pupil_area

def raster_plot(gs):

    X_embedding, aligned_saccades, pupil_x, pupil_y, exact_saccades, pupil_timestamp, spks_timestamp, motSVD, blink_idx, neural_frames_list, cam_frames_list, pupil_area = get_behavioral_data("dark")
    
    nested_gs = GridSpecFromSubplotSpec(3, 1, height_ratios=[.7, .2, .1], subplot_spec=gs, hspace=0.)

    vline = neural_frames_list[0]
    vline_cam = cam_frames_list[0]
    vline_width = .5

    neural_xmin = vline - (2*60*3)
    neural_xmax = vline + (4*60*3)
    idx_after = 4*60*3
    neural_xlim = [neural_xmin, neural_xmax]
    pupil_xmin = np.argmin(np.abs(pupil_timestamp.item()['blk1'] - spks_timestamp.item()['blk1'][neural_xmin]))
    pupil_xmax = np.argmin(np.abs(pupil_timestamp.item()['blk2'] - spks_timestamp.item()['blk2'][idx_after]))
    # session end then restart timestamp
    second_session_start = np.argmin(np.abs(pupil_timestamp.item()['blk2'] - spks_timestamp.item()['blk2'][0])).astype(int)
    pupil_xlim = [pupil_xmin+second_session_start, vline_cam + pupil_xmax]

    # Nan out values during blink times
    pupil_x_nan = pupil_x.copy()
    pupil_y_nan = pupil_y.copy()
    pupil_x_nan[blink_idx] = np.nan
    pupil_y_nan[blink_idx] = np.nan
    # Also mask values before the vline
    pupil_x_nan[:vline_cam+second_session_start] = np.nan
    pupil_y_nan[:vline_cam+second_session_start] = np.nan    

    # Plot rastermap sorted by pupil area correlation
    ax0 = plt.subplot(nested_gs[0])
    ax0.imshow(X_embedding, vmin=0, vmax=1, cmap="gray_r", aspect="auto")
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.set_ylim(X_embedding.shape[0],0)
    ax0.set_xlim(neural_xlim)
    for s in aligned_saccades:
        ax0.axvline(s+vline, color=(0,1,0), linestyle='dashed', lw=vline_width)
    remove_spines(ax0)
    # Add y-axis scale bar for rastermap
    bin_size = 200
    scale_bar_length = 10
    scale_bar_start = ax0.get_ylim()[0] 
    scale_bar_end = scale_bar_start - scale_bar_length
    # orient the text vertically
    ax0.plot([ax0.get_xlim()[0]-10, ax0.get_xlim()[0]-10], [scale_bar_start, scale_bar_end], 'k-', lw=2, clip_on=False)
    ax0.text(ax0.get_xlim()[0]-35, scale_bar_start - 35 + scale_bar_length/2, f'{scale_bar_length*bin_size} neurons', ha='left', va='center', rotation='vertical')
    ax0.axvline(vline, color='r', linestyle='dashed', lw=.5)

    # Plot pupil x and y
    pupil_x_color = 'darkgreen'
    pupil_y_color = 'lightgreen'
    ax1 = plt.subplot(nested_gs[1])
    ax1.plot(pupil_x_nan, color=pupil_x_color, label='Pupil X', lw=.5)
    ax1.plot(pupil_y_nan, color=pupil_y_color, label='Pupil Y', lw=.5)
    # Plot saccades
    for s in exact_saccades:
        if s[0] in blink_idx or s[0].astype(int) < second_session_start:
            continue
        ax1.axvline(s[0].astype(int)+vline_cam, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
    ax1.set_xlim(pupil_xlim)
    #legend = ax1.legend(facecolor='none', frameon=False, bbox_to_anchor=(-.002, .4), fontsize=LABEL_FONTSIZE-4)
    legend = ax1.legend(facecolor='none', frameon=False, bbox_to_anchor=(0.05, .52, .1, .1), fontsize=LABEL_FONTSIZE-4)
    leg_colors = [pupil_x_color, pupil_y_color]
    for leg in legend.get_texts():
        leg.set_color(leg_colors.pop(0))
    # remove legend lines 
    for leg_line in legend.get_lines():
        leg_line.set_linewidth(0)
    ax1.set_yticks([])
    ax1.set_xticks([])
    remove_spines(ax1)
    # Add x-axis scale bar for pupil area, x and y, rastermap on the right
    duration = 10 # seconds
    behav_sampling_rate = np.mean(np.diff(pupil_timestamp.item()['blk2'])) * 24 * 3600 * 1000
    scale_bar_length = behav_sampling_rate * duration
    scale_bar_start = ax1.get_xlim()[1] - 500
    scale_bar_end = scale_bar_start + scale_bar_length
    ax1.plot([scale_bar_start, scale_bar_end], [ax1.get_ylim()[0]+2, ax1.get_ylim()[0]+2], 'k-', lw=2, clip_on=False)
    ax1.text(scale_bar_start + scale_bar_length/2, ax1.get_ylim()[0]+2.5, f'{int(duration)} s'.format(str(duration)), ha='center', va='bottom', fontsize=LABEL_FONTSIZE-3)
    # Add y-axis scale bar for pupil x and y
    scale_bar_length = 3
    scale_bar_start = ax1.get_ylim()[1] - 17
    scale_bar_end = scale_bar_start + scale_bar_length
    ax1.plot([ax1.get_xlim()[0]+5, ax1.get_xlim()[0]+5], [scale_bar_start, scale_bar_end], 'k-', lw=1.5, clip_on=False)
    ax1.text(ax1.get_xlim()[0]-(0.035*np.diff(ax1.get_xlim())), scale_bar_start + scale_bar_length/2, f'{scale_bar_length} px', ha='left', va='center', fontsize=LABEL_FONTSIZE-3)
    ax1.axvline(vline_cam+second_session_start, color='r', linestyle='dashed', lw=.5)

    # Plot pupil area trace
    ax2 = plt.subplot(nested_gs[2])
    line, = ax2.plot(pupil_area, label='Pupil\narea', color='orange', lw=2)
    # Plot saccades
    for s in exact_saccades:
        if s[0] in blink_idx or s[0].astype(int) < second_session_start:
            continue
        ax2.axvline(s[0].astype(int)+vline_cam, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
    #line.set_path_effects([pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    legend=ax2.legend(loc='center left', bbox_to_anchor=(-0.15, 0.55), facecolor='none')
    legend.get_texts()[0].set_color('orange') 
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim(pupil_xlim)
    ax2.axvline(vline_cam, color='r', linestyle='dashed')
    remove_spines(ax2)

    # Add y-axis scale bar for pupil area
    scale_bar_length = 500
    scale_bar_start = ax1.get_ylim()[0] + 10
    scale_bar_end = scale_bar_start + scale_bar_length
    ax1.plot([ax1.get_xlim()[0]+5, ax1.get_xlim()[0]+5], [scale_bar_start, scale_bar_end], 'k-', lw=1)
    ax1.text(ax1.get_xlim()[0]+10, scale_bar_start + scale_bar_length/2, f'{scale_bar_length} AU', ha='left', va='center')

    return 
"""
"""
def raster_plot_dark(gs):

    dbs = pdh.carbachol_in_dark() 
    base_folder = '/home/asyeda/pupil_analysis/dark_recordings/'
    mname = 'LZ14'
    datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]

    # Check if file exists 
    if os.path.exists(f"{base_folder}/{mname}_{datexp}_rasterdata_udcnv.npz"):
        data = np.load(f"{base_folder}/{mname}_{datexp}_rasterdata_udcnv.npz", allow_pickle=True)
        X_embedding = data['X_embedding']
        pupil_x = data['pupil_x']
        pupil_y = data['pupil_y']
        pupil_area = data['pupil_area']
        neural_frames_list = data['neural_frames_list']
        tneural = data['tneural']
        saccade_data = np.load('/home/asyeda/pupil_analysis/saccade_times.npy', allow_pickle=True).item()
        aligned_saccades = []
        # Get saccades
        for i, block in enumerate(saccade_data["dark"][mname].keys()):
            _, a_sac = get_aligned_saccade_times(saccade_data["dark"][mname][block], data['tneural'].item()['blk{}'.format(block)])
            a_sac = a_sac.astype(int)
            if i==0:
                aligned_saccades.append(a_sac)
        aligned_saccades = np.hstack(aligned_saccades)

    else:
        saccade_data = np.load('/home/asyeda/pupil_analysis/saccade_times.npy', allow_pickle=True).item()
        mouse_dbs = [db for db in dbs if db['mname'] == mname]
        blk = mouse_dbs[0]['blk']
        npz_data = np.load(f"{base_folder}/{mname}_{datexp}_{blk}.npz", allow_pickle=True)

        model = Rastermap().fit(zscore(npz_data['spks'], axis=1).copy())
        y = model.embedding # neurons x 1
        isort = model.isort
        pupil_x, pupil_y, pupil_area = [], [], []
        spks = []
        for block_idx in range(2):
            block = list(npz_data['tcam'].item().keys())[block_idx]
            tcam = npz_data['tcam'].item()[block]
            tneural = npz_data['tneural'].item()[block]
            # Resample behavior data to neural timestamps
            pupil_x.append(helper_functions.interp_value(helper_functions.segment_data(npz_data['pupil_x_polar'], npz_data['cam_frames_list'], block_idx), tcam, tneural))
            pupil_y.append(helper_functions.interp_value(helper_functions.segment_data(npz_data['pupil_y_polar'], npz_data['cam_frames_list'], block_idx), tcam, tneural))
            pupil_area.append(helper_functions.interp_value(helper_functions.segment_data(npz_data['pupil_area_small'], npz_data['cam_frames_list'], block_idx), tcam, tneural))
            spks.append(helper_functions.segment_data(npz_data['spks'].T, npz_data['neural_frames_list'], block_idx).T)

        pupil_x = np.concatenate(pupil_x, axis=0)
        pupil_y = np.concatenate(pupil_y, axis=0)
        pupil_area = np.concatenate(pupil_area, axis=0)
        spks = np.concatenate(spks, axis=1)
        neural_frames_list = npz_data['neural_frames_list']
        cam_frames_list = npz_data['cam_frames_list']
        tneural = npz_data['tneural']
        X_embedding = zscore(utils.bin1d(spks[isort], bin_size=200, axis=0), axis=1)

        np.savez(f"{base_folder}/{mname}_{datexp}_rasterdata_udcnv.npz", X_embedding=X_embedding, pupil_x=pupil_x, pupil_y=pupil_y, pupil_area=pupil_area, neural_frames_list=npz_data['neural_frames_list'], cam_frames_list=npz_data['cam_frames_list'], tneural=npz_data['tneural'], tcam=npz_data['tcam'])

    # Get blinks for the first carbachol recording only
    blink_data = np.load('/home/asyeda/pupil_analysis/blink_times.npy', allow_pickle=True).item()
    neural_blink_times, neural_blink_idx = get_aligned_blink_times(blink_data['dark'][mname]['2'], tneural.item()['blk2'])

    # Plot
    nested_gs = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs, hspace=0.15, height_ratios=[.8, .1, .2, .25])
    vline = neural_frames_list[0]
    start = vline - (2*60*3)
    end = vline + (6*60*3)
    xlim = [start,end]
    vline_width = .4

    # Colors for pupil x and y
    pupil_y_color = 'lightgreen'
    pupil_x_color = 'darkgreen'

    neural_blink_idx = neural_blink_idx + vline
    pupil_x_nan = pupil_x.copy()
    pupil_y_nan = pupil_y.copy()
    # Mask pupil x and y values during blinks
    pupil_x_nan[neural_blink_idx] = np.nan
    pupil_y_nan[neural_blink_idx] = np.nan

    # Plot rastermap sorted by pupil area correlation
    ax0 = plt.subplot(nested_gs[0])
    ax0.imshow(X_embedding, vmin=0, vmax=1, cmap="gray_r", aspect="auto")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlim(xlim)
    #ax0.vlines(np.cumsum(npz_data['neural_frames_list']), ymin=0, ymax=X_embedding.shape[0], colors='r', linestyles='dashed')
    ax0.set_ylim(X_embedding.shape[0],0)
    ax0.axvline(vline, color='r', linestyle='dashed')
    for s in aligned_saccades:
        ax0.axvline(s+vline, color=(0,1,0), linestyle='dashed', lw=vline_width)
    remove_spines(ax0)
    # Add y-axis scale bar for rastermap
    bin_size = 200
    scale_bar_length = 10
    scale_bar_start = ax0.get_ylim()[0] 
    scale_bar_end = scale_bar_start - scale_bar_length
    # orient the text vertically
    ax0.plot([ax0.get_xlim()[0]-10, ax0.get_xlim()[0]-10], [scale_bar_start, scale_bar_end], 'k-', lw=2, clip_on=False)
    ax0.text(ax0.get_xlim()[0]-45, scale_bar_start - 35 + scale_bar_length/2, f'{scale_bar_length*bin_size} neurons', ha='left', va='center', rotation='vertical')

    # Plot pupil area trace
    ax1 = plt.subplot(nested_gs[1])
    line, = ax1.plot(pupil_area, label='Pupil\narea', color='olive', lw=1.2)
    for s in aligned_saccades:
        if s+vline <= vline:
            continue
        ax1.axvline(s+vline, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
    #line.set_path_effects([pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    legend=ax1.legend(loc='center left', bbox_to_anchor=(-0.15, 0.55), facecolor='none')
    legend.get_texts()[0].set_color('olive') 
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_xlim(xlim)
    ax1.axvline(vline, color='r', linestyle='dashed')
    remove_spines(ax1)

    # Add y-axis scale bar for pupil area
    scale_bar_length = 500
    scale_bar_start = ax1.get_ylim()[0] + 10
    scale_bar_end = scale_bar_start + scale_bar_length
    #ax1.plot([ax1.get_xlim()[0]+5, ax1.get_xlim()[0]+5], [scale_bar_start, scale_bar_end], 'k-', lw=1)
    #ax1.text(ax1.get_xlim()[0]+10, scale_bar_start + scale_bar_length/2, f'{scale_bar_length} AU', ha='left', va='center')

    # Plot pupil x trace
    # Mask data before the red dashed line (vline)
    masked_pupil_x = np.ma.masked_where(np.arange(len(pupil_x_nan)) < vline, pupil_x_nan)
    masked_pupil_y = np.ma.masked_where(np.arange(len(pupil_y_nan)) < vline, pupil_y_nan)

    ax2 = plt.subplot(nested_gs[2])
    ax2.plot(masked_pupil_y, label='Elevation', color=pupil_y_color, lw=1.2)
    ax2.plot(masked_pupil_x, label='Azimuth', color=pupil_x_color, lw=1.2)
    for s in aligned_saccades:
        if s+vline <= vline:
            continue
        ax2.axvline(s+vline, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(xlim)
    legend=ax2.legend(loc='center left', bbox_to_anchor=(-0.15, 0.45),facecolor='none')
    legend.get_texts()[0].set_color(pupil_y_color)  
    legend.get_texts()[1].set_color(pupil_x_color)  
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)
    #ax2.set_ylim(np.nanpercentile([pupil_x,pupil_y], .1), np.nanpercentile([pupil_x,pupil_y], 99.99))
    ax2.axvline(vline, color='r', linestyle='dashed')
    remove_spines(ax2)
    # Add y-axis scale bar for pupil x and y
    scale_bar_length = 10
    scale_bar_start = ax2.get_ylim()[0] + 25
    scale_bar_end = scale_bar_start + scale_bar_length
    ax2.plot([vline-17, vline-17], [scale_bar_start, scale_bar_end], 'k-', lw=1, clip_on=False, zorder=10)
    ax2.text(vline-65, scale_bar_start + scale_bar_length/2, f'{scale_bar_length}$^\circ$', ha='left', va='center', clip_on=False)

    # Add x-axis scale bar for pupil area, x and y, rastermap on the right
    duration = 10 # seconds
    scale_bar_length = 3 * duration
    scale_bar_start = ax1.get_xlim()[1] - 50
    scale_bar_end = scale_bar_start + scale_bar_length
    ax1.plot([scale_bar_start, scale_bar_end], [ax1.get_ylim()[1]-200, ax1.get_ylim()[1]-200], 'k-', lw=1)
    ax1.text(scale_bar_start + scale_bar_length/2, ax1.get_ylim()[1]-25, f'{int(duration)} s'.format(str(duration)), ha='center', va='bottom', clip_on=False)

    # Under vline add text annotation carbachol added
    ax2.text(vline+10, ax2.get_ylim()[0]-15, 'Carbachol added', ha='center', va='bottom', fontsize=10, color='r', clip_on=False)

    # Position axes
    pos = ax0.get_position().bounds
    ax0.set_position([pos[0], pos[1]-0.09, pos[2], pos[3]*1.5])
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0], pos[1]-0.12, pos[2], pos[3]*1.5])
    pos = ax2.get_position().bounds
    ax2.set_position([pos[0], pos[1]-0.15, pos[2], pos[3]*1.5])
    """
def get_aligned_blink_times(blink_data, spks_timestamps):
    blink_start_t = blink_data[:,0]
    blink_end_t = blink_data[:,1]
    neural_blink_start_idx = np.zeros(len(blink_start_t))
    neural_blink_end_idx = np.zeros(len(blink_end_t))
    for i, t in enumerate(blink_start_t):
        try:
            start_time = np.argmin(np.abs(spks_timestamps - t))
            neural_blink_start_idx[i] = start_time
        except:
            continue
    for i, t in enumerate(blink_end_t):
        try:
            end_time = np.argmin(np.abs(spks_timestamps - t))
            neural_blink_end_idx[i] = end_time
        except:
            continue
    # return indices from start-end for blink duration
    neural_blink_idx = []
    for s, e in zip(neural_blink_start_idx, neural_blink_end_idx):
        if e > s:
            neural_blink_idx = np.append(neural_blink_idx, np.arange(s, e+1))
        else:
            neural_blink_idx = np.append(neural_blink_idx, s)
    neural_blink_idx = np.unique(neural_blink_idx.astype(int))
    neural_blink_times = spks_timestamps[neural_blink_idx]
    return neural_blink_times, neural_blink_idx

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

def remove_spines(ax):
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)

def plot_varexp_lineplot(gs):

    base_folder = '/home/asyeda/pupil_analysis/dark_recordings'
    nonlinear_results = np.load(f"{base_folder}/nonlinear_dark_results_udcnv.npy", allow_pickle=True).item()

    # Plot line plot
    input_labels = ['Pupil position', 'Face w/o eye','Face w/o eye\n+ pupil position']
    alpha = 0.05  # significance level
    alpha2 = 0.001  # significance level
    varexp_data = []  # Collect normalized variance for all animals

    # Collect normalized variance explained for all animals
    for mname in nonlinear_results:
        varexp_norms = []
        for name in ['pupil', 'face_eye', 'face_eye_pupil']:
            value = (nonlinear_results[mname][name]['vexp_after_carbachol']) * 100 # vexp_after_carbachol vexp_before_carbachol
            #print(mname, name, value)
            varexp_norms.append(value)
        varexp_data.append(varexp_norms)

    # Convert to NumPy array for easier manipulation
    varexp_data = np.array(varexp_data)

    # Remove the singleton dimension
    varexp_data = np.squeeze(varexp_data)  # Shape becomes (4, 4)

    # Perform ANOVA
    anova_pval = f_oneway(*[varexp_data[:, i] for i in range(varexp_data.shape[1])]).pvalue

    # Pairwise comparisons (Bonferroni corrected)
    pairwise_pvals = []
    for i in range(len(input_labels)):
        for j in range(i + 1, len(input_labels)):
            _, pval = ttest_rel(varexp_data[:, i], varexp_data[:, j])
            pairwise_pvals.append((i, j, pval))

    # print results for paper
    means = varexp_data.mean(axis=0)
    stds = varexp_data.std(axis=0, ddof=1)
    print("=== Mean ± SD of normalized variance explained (%): ===")
    for label, m, s in zip(input_labels, means, stds):
        print(f"{label:15s}: {m:5.1f}\% $\pm$ {s:4.1f}\%")

    # Plot configuration
    ax = plt.subplot(gs)

    for i, animal_data in enumerate(varexp_data):
        # ax.plot(animal_data, 'o-', label=f"Mouse {list(nonlinear_results.keys())[i]}", markersize=3, color=colors[i])
        ax.plot(animal_data, '-', label=f"Mouse {i+1}", markersize=3, color='grey', lw=1, alpha=.7)
    # Plot average line across all mice
    ax.plot(np.mean(varexp_data, axis=0), '-', label='Average', markersize=3, color='black', lw=3.5)

    ax.set_xticks(np.arange(len(input_labels)), input_labels, fontsize=LABEL_FONTSIZE-2, rotation=35, ha='right', va='center_baseline')
    # change color of x ticks
    xtick_colors = ['green', 'darkorange', 'red']
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(xtick_colors[i])
    ax.set_ylabel("Neural PC variance explained (%)")

    # Add inset axis for significance annotations above the plot
    axin = ax.inset_axes([0, 1.03, 1, 0.12])  # Position the inset axis above the main axis
    axin.axis('off')  # Hide the inset axis

    # Annotate significant results
    prev_line_y = 0  # Initialize previous line y-coordinate
    for i in range(varexp_data.shape[1]):
        for j in range(i + 1, varexp_data.shape[1]):
            corrected_pval = pairwise_pvals.pop(0)
            if corrected_pval[2] < alpha:  # Use the third element of the tuple
                x = [i, j]  # List for x-coordinates
                if prev_line_y == 0:
                    y = varexp_data[:, [i, j]].max()
                else:
                    y = prev_line_y + 1.5
                axin.plot([i, j], [y, y], 'k-', lw=.7)  # Use lists for plotting horizontal line
                if corrected_pval[2] < alpha2:
                    axin.text((i + j) / 2, y-.5, '***', ha='center')
                else:
                    axin.text((i + j) / 2, y-.5, '*', ha='center')
                prev_line_y = y  # Update previous line y-coordinate
            else: # add n.s
                if prev_line_y == 0:
                    y = varexp_data[:, [i, j]].max()
                else:
                    y = prev_line_y + 1.5
                axin.plot([i, j], [y, y], 'k-', lw=.7)  # Use lists for plotting horizontal line
                axin.text((i + j) / 2, y + 0.02, 'n.s.', ha='center', fontsize=8)
                prev_line_y = y

    ax.set_ylim(0, 35)
    ax.set_yticks([0, 15, 30])

    # Adjust position
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+.035, pos[1]+.15, pos[2]*.7, pos[3]*.8])

    # Add inset axis for another datapoint
    axin = ax.inset_axes([1.2, 0.0, 0.2, 1])  # Position the inset axis in the bottom right corner
    before_varexp_data = []  # Collect normalized variance for all animals
    # Collect normalized variance explained for all animals
    for mname in nonlinear_results:
        varexp_norms = []
        for name in ['face_eye']:
            value = (nonlinear_results[mname][name]['vexp_before_carbachol']) * 100 # vexp_after_carbachol vexp_before_carbachol
            varexp_norms.append(value)
        before_varexp_data.append(varexp_norms)
    before_varexp_data = np.array(before_varexp_data)
    before_varexp_data = np.squeeze(before_varexp_data)  # Shape becomes (4, 4)

    # print before carbachol results for paper
    mean_before = np.mean(before_varexp_data)
    std_before = np.std(before_varexp_data, ddof=1)
    print("=== Mean ± SD of normalized variance explained (%), before carbachol: ===")
    print(f"{mean_before:5.1f}\% $\pm$ {std_before:4.1f}\%")

    for i, animal_data in enumerate(before_varexp_data):
        axin.scatter(1, animal_data, color='grey', alpha=.5, marker='.')
    # Plot average line across all mice
    axin.scatter(1, np.mean(before_varexp_data, axis=0), color='black', marker='.')  # Larger marker size for clarity    #axin.set_xticks(1, 'Face w/o eye\n(before carbachol)', fontsize=12, rotation=40, ha='right', va='center_baseline')
    axin.set_xticks([1], ['Face w/o eye\n(before carbachol)'], fontsize=LABEL_FONTSIZE-2, rotation=40,
                     ha='right', va='center_baseline', y=-0.05)
    axin.set_ylim(0, 35)
    axin.set_yticks([])
    axin.spines['left'].set_visible(False)

    # Do pairwise test b/w varexp_data (face_eye) and before_varexp_data (face_eye)
    # Pairwise comparisons (Bonferroni corrected)
    _, pval = ttest_rel(varexp_data[:, 1], before_varexp_data)
    print("Pairwise p-value between Face w/o eye (after carbachol) and Face w/o eye (before carbachol):", pval)
    # y-coordinate for the significance line (in data units)
    y_line = 32  

    x_main = 1
    y_line = 35  # y-position for the bar (adjust if needed)

    # Convert the left point (main axis, Pupil position) into figure coordinates
    fig = ax.figure
    x0, y0 = ax.transData.transform((x_main, y_line))
    x0f, y0f = fig.transFigure.inverted().transform((x0, y0))

    # Convert the right point (inset axis, Face w/o eye (before carbachol)) into figure coordinates
    x1, y1 = axin.transData.transform((1, y_line))  # inset tick is at x=1
    x1f, y1f = fig.transFigure.inverted().transform((x1, y1))

    # Draw a line spanning across the two axes
    fig.lines.append(plt.Line2D([x0f, x1f], [y0f, y1f], transform=fig.transFigure, color='k', lw=0.7))

    if pval < alpha:
        if pval < alpha2:
            fig.text(0.5*(x0f+x1f), y1f, '***', ha='center', fontsize=8)
        else:
            fig.text(0.5*(x0f+x1f), y1f, '*', ha='center', fontsize=8)
    else: # Add "n.s." label above the line
        fig.text(0.5*(x0f+x1f), y1f, 'n.s.', ha='center', fontsize=8)


def plot_varexp_curve(gs):

    base_folder = '/home/asyeda/pupil_analysis/dark_recordings'
    dbs = pdh.carbachol_in_dark()
    nonlinear_results = np.load(f"{base_folder}/nonlinear_dark_results_udcnv.npy", allow_pickle=True).item()
    mouse_list = np.unique([entry['mname'] for entry in dbs])

    # Define the figure with specified size and higher DPI for better quality
    axs = plt.subplot(gs)
    cumsum_varexp = {} # cumulative sum of variance explained for each behavior type

    # Iterate over each mouse in the mouse list
    for mi, mname in enumerate(mouse_list):
        vexp_list = []
        vexp_pc_list = []
        
        # Extract values for each measurement
        input_labels = []
        for behav_input in nonlinear_results[mname].keys():
            if behav_input == 'fullface' or behav_input == 'pupil_area':
                continue
            if 'pupil' in behav_input and 'dark' not in base_folder:
                vexp = nonlinear_results[mname][behav_input]['vexp_before_carbachol']   # vexp_after_carbachol vexp_before_carbachol
                vexp_pc = nonlinear_results[mname][behav_input]['vexp_pc_before_carbachol'] #  vexp_pc_after_carbachol vexp_pc_before_carbachol
                vpcs = nonlinear_results[mname][behav_input]['vpcs_before_carbachol'] # vpcs_after_carbachol  vpcs_before_carbachol
            else:
                vexp = nonlinear_results[mname][behav_input]['vexp_after_carbachol']   # vexp_after_carbachol vexp_before_carbachol
                vexp_pc = nonlinear_results[mname][behav_input]['vexp_pc_after_carbachol'] #  vexp_pc_after_carbachol vexp_pc_before_carbachol
                vpcs = nonlinear_results[mname][behav_input]['vpcs_after_carbachol'] # vpcs_after_carbachol  vpcs_before_carbachol
            vexp_list.append(vexp)
            vnorm = vpcs / vpcs.sum()
            vexp_pc_list.append(vexp_pc)
            input_labels.append(behav_input)
        
        # Convert lists to numpy arrays
        vexp_array = np.array(vexp_list)
        vexp_pc_array = np.array(vexp_pc_list)
        
        # computing cumulative explained variance
        vec_cumsum = np.cumsum(vexp_pc_array * vnorm, axis=1).T
        cumsum_varexp[mname] = vec_cumsum

    # Plot average across all mice
    vexp_array = np.array([cumsum_varexp[mname] for mname in mouse_list])
    mean_vexp_array = np.mean(vexp_array, axis=0)
    std_vexp_array = np.std(vexp_array, axis=0)
    # compute standard error of the mean
    sem_vexp_array = std_vexp_array / np.sqrt(vexp_array.shape[0])
    colors = ['green', 'darkorange', 'red']
    labels = ['Pupil position', 'Face w/o eye', 'Face w/o eye +\npupil position']
    ordered_labels = ['pupil', 'face_eye', 'face_eye_pupil']
    ls_list = ['--', '-.', ':']
    for j in range(mean_vexp_array.shape[1]):
        idx = input_labels.index(ordered_labels[j])
        axs.plot(
            np.arange(1, 129),
            mean_vexp_array[:, idx] * 100,
            linewidth=2,
            label=labels[j],
            color=colors[j],
            ls=ls_list[j]
        )
    for std_idx in range(sem_vexp_array.shape[1]):
        idx = input_labels.index(ordered_labels[std_idx])
        axs.fill_between(np.arange(1, 129), (mean_vexp_array[:,idx] - sem_vexp_array[:,idx])*100, 
                         (mean_vexp_array[:,idx] + sem_vexp_array[:,idx])*100, alpha=0.03, color=colors[std_idx])

    axs.set_xscale('log')
    axs.set_xlabel("Neural activity principal\ncomponent", fontsize=LABEL_FONTSIZE-1)
    axs.set_ylabel("% Cumulative variance explained\n(test data)")
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_xticklabels([1, 10, 100])
    axs.set_xticks([1, 10, 100])
    axs.set_ylim([0, 30])
    axs.set_yticklabels([0, 10, 20, 30])
    axs.set_yticks([0, 10, 20, 30])

    # change the legends to remove lines and have same color as the plot
    legend=axs.legend(loc='center left', bbox_to_anchor=(0.25, 0.45), fontsize=LABEL_FONTSIZE-2, frameon=False,
                      handlelength=0)
    for j in range(len(legend.get_texts())):
        legend.get_texts()[j].set_color(colors[j])
    legend.get_frame().set_facecolor('none')
    
    # Adjust position
    pos = axs.get_position().bounds
    axs.set_position([pos[0]+.035, pos[1]+.085, pos[2], pos[3]*.85])
    
    # Perform a paired test between the 1st PC of face_eye and pupil
    face_eye_data = vexp_array[:, :, input_labels.index('face_eye')]
    pupil_data = vexp_array[:, :, input_labels.index('pupil')]
    print("face_eye_data:", face_eye_data[:,0])
    print("pupil_data:", pupil_data[:,0])
    tval, pval = ttest_rel(face_eye_data[:,0], pupil_data[:,0])
    print(f"ttest test between 1st PC of face_eye and pupil stat: {tval}: p-value = {pval}")
    
    # Perform test between the last PC of face_eye and pupil
    print("face_eye_data:", face_eye_data[:,-1])
    print("pupil_data:", pupil_data[:,-1])
    tval, pval = ttest_rel(face_eye_data[:,-1], pupil_data[:,-1])
    # print numer
    print(f"ttest test between last PC of face_eye and pupil stat: {tval}: p-value = {pval}")

def in_any_range(x, ranges):
    return any(start <= x <= end for start, end in ranges)

def ranges_from_indices(indices):
    indices = sorted(map(int, indices))  # ensure sorted + cast to int
    ranges = []
    for _, g in groupby(enumerate(indices), lambda x: x[1] - x[0]):
        g = list(g)
        ranges.append((int(g[0][1]), int(g[-1][1])))
    return ranges


from itertools import groupby

def ranges_from_indices(indices):
    indices = sorted(map(int, indices))  # ensure sorted + cast to int
    ranges = []
    for _, g in groupby(enumerate(indices), lambda x: x[1] - x[0]):
        g = list(g)
        ranges.append((int(g[0][1]), int(g[-1][1])))
    return ranges

def in_any_range(x, ranges):
    return any(start <= x <= end for start, end in ranges)


def plot_fve_prepost_carbachol(gs):
    dbs = pdh.carbachol_in_dark()
    base_folder = '/home/asyeda/pupil_analysis/dark_recordings'
    mouse_list = np.unique([pd['mname'] for pd in dbs])    
    pre_fve_all = []
    post_fve_all = []

    if os.path.exists(f"{base_folder}/carbachol_pc_fve.npz"):
        carbachol_pc_fve = np.load(f"{base_folder}/carbachol_pc_fve.npz", allow_pickle=True)
        pre_fve_all = carbachol_pc_fve['pre_fve_all']
        post_fve_all = carbachol_pc_fve['post_fve_all']
    else:
        for mname in mouse_list:
            datexp = [db['datexp'] for db in dbs if db['mname'] == mname][0]
            blocks = [db['blk'] for db in dbs if db['mname'] == mname]
            _, _, _, tcams, tneurals, cumframes_neural, cumframes_cam, npz_data = nonlinear_refactored.load_data(base_folder, mname, datexp)

            # Load spks using the new deconvolution method (udcnv)
            spks_udcnv_file = "/media/Atika/neural_data_udcnv/{}_{}_1_spks.npz".format(mname, datexp)
            spks = np.load(spks_udcnv_file)['spks']
            
            carbachol_block = list(npz_data['tcam'].item().keys())[1]
            # Pre-carbachol data
            pre_carbachol_spks = helper_functions.segment_data(spks.T, npz_data['neural_frames_list'], 0).T
            pre_carbachol_spks = nonlinear_refactored.zscore_neural_data(pre_carbachol_spks)

            # Post-carbachol data
            post_carbachol_spks = helper_functions.segment_data(spks.T, npz_data['neural_frames_list'], 1).T
            post_carbachol_spks = nonlinear_refactored.zscore_neural_data(post_carbachol_spks)

            # Get PCA of neural data from pre-carbachol period 
            num_neural_frames = 20 * 60 * 3 # 20 mins before carbachol, 3 Hz sampling rate
            pca = PCA(n_components=128)
            pca.fit(pre_carbachol_spks[:, -num_neural_frames:].T)
            # baseline variances from PCA fit
            baseline_var = pca.explained_variance_

            num_neural_frames = 10 * 60 * 3 # 10 mins after carbachol, 3 Hz sampling rate
            # --- First 10 min (pre-carchachol) ---
            pre_data = pre_carbachol_spks[:, :num_neural_frames]
            pre_proj = pca.transform(pre_data.T)
            pre_var = np.var(pre_proj, axis=0)
            
            # --- First 10 min (post-carchachol) ---
            post_data = post_carbachol_spks[:, :num_neural_frames]
            post_proj = pca.transform(post_data.T)
            post_var = np.var(post_proj, axis=0)
            
            # --- Normalize by total baseline variance ---
            pre_fve = pre_var / baseline_var.sum()
            post_fve = post_var / baseline_var.sum()

            pre_fve_all.append(pre_fve)
            post_fve_all.append(post_fve)

        pre_fve_all = np.array(pre_fve_all)
        post_fve_all = np.array(post_fve_all)
        np.savez(f"{base_folder}/carbachol_pc_fve.npz", pre_fve_all=pre_fve_all, post_fve_all=post_fve_all)
    # Plot
    color_pre = "#377eb8"   # medium blue
    color_post = "brown"

    ax = plt.subplot(gs)
    ax.plot(np.arange(1, 129), np.cumsum(pre_fve_all.mean(axis=0)), label='Pre-carbachol', color=color_pre)
    ax.plot(np.arange(1, 129), np.cumsum(post_fve_all.mean(axis=0)), label='Post-carbachol', color=color_post)
    # plot sem
    ax.fill_between(np.arange(1, 129), 
                    np.cumsum(pre_fve_all.mean(axis=0) - pre_fve_all.std(axis=0)/np.sqrt(len(pre_fve_all))), 
                    np.cumsum(pre_fve_all.mean(axis=0) + pre_fve_all.std(axis=0)/np.sqrt(len(pre_fve_all))), 
                    color=color_pre, alpha=0.2)
    ax.fill_between(np.arange(1, 129),
                    np.cumsum(post_fve_all.mean(axis=0) - post_fve_all.std(axis=0)/np.sqrt(len(post_fve_all))), 
                    np.cumsum(post_fve_all.mean(axis=0) + post_fve_all.std(axis=0)/np.sqrt(len(post_fve_all))), 
                    color=color_post, alpha=0.2)
    ax.legend(frameon=False, fontsize=LABEL_FONTSIZE-2)
    # change ylim to 0-.3 with ticks at 0, .15, .3
    ax.set_ylim(0, 0.25)
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_yticklabels([0, 10, 20], fontsize=LABEL_FONTSIZE-2)
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels([1, 10, 100], fontsize=LABEL_FONTSIZE-2)

    # remove legend line and only use text which is colored
    legend = ax.get_legend()
    for line in legend.get_lines():
        line.set_linewidth(0)
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor('none')
    for text, color in zip(legend.get_texts(), [color_pre, color_post]):
        text.set_color(color)

    ax.set_xlabel('Principal component')
    ax.set_ylabel('% Cumulative variance explained')
    #ax.set_title('Carbachol and neural variability', y=1.04)

    # Add an inset axis showing scatter points connected by lines. x-axis is pre and post, y-axis is total variance explained by 128 PCs
    axin = ax.inset_axes([1.15, 0.0, 0.2, 1])  # Position the inset axis in the bottom right corner
    pre_total_fve = np.cumsum(pre_fve_all, axis=1)[:, -1] * 100
    post_total_fve = np.cumsum(post_fve_all, axis=1)[:, -1] * 100   
    for i in range(len(pre_total_fve)):
        axin.plot([0, 1], [pre_total_fve[i], post_total_fve[i]], color='grey', alpha=0.5, marker='o',
                   markerfacecolor='black', markeredgecolor='black', markersize=4, lw=1, clip_on=False)
    #axin.scatter(0, np.mean(pre_total_fve), color='black', marker='o', s=50, clip_on=False)  # Larger marker size for clarity
    #axin.scatter(1, np.mean(post_total_fve), color='black', marker='o', s=50, clip_on=False)  # Larger marker size for clarity
    axin.set_xticks([0, 1])
    axin.set_xticklabels(['Pre-carbachol', 'Post-carbachol'], fontsize=LABEL_FONTSIZE-3, rotation=90, 
                         ha='right', va='center_baseline')
    # use same color for xticklabels as the plot lines
    axin.get_xticklabels()[0].set_color(color_pre)
    axin.get_xticklabels()[1].set_color(color_post)
    axin.set_ylabel('% Total variance explained', fontsize=LABEL_FONTSIZE-1)
    axin.set_ylim(0, 25)
    axin.set_yticks([])
    axin.spines['left'].set_visible(False)
    axin.spines['top'].set_visible(False)
    axin.spines['right'].set_visible(False)
    # Do paired t-test
    _, pval = ttest_rel(pre_total_fve, post_total_fve)
    print("Paired t-test p-value between pre and post carbachol total variance explained by 128 PCs:", pval)
    alpha = 0.05
    # y-coordinate for the significance line (in data units)
    y_line = 25

    if pval < alpha:
        # Significant: draw line in main axes (will be visible in inset if it overlaps)
        axin.plot([-.1, 1.1], [y_line, y_line], 'k-', lw=0.7)
        axin.text(0.5, y_line + 1, '*', ha='center', fontsize=8)
    else:
        x_main = 0.5
        axin.plot([-.1, 1.1], [y_line, y_line], 'k-', lw=0.7)
        axin.text(0.5, y_line + .5, 'n.s.', ha='center', fontsize=8)

    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]+.13, pos[2]*.7, pos[3]*.85])

def raster_plot_predictions(fig, gs, recording_type="dark"):

    if recording_type == 'dark':
        rastermap_out = np.load("/home/asyeda/pupil_analysis/dark_recordings/rastermap_out_dark_udcnv.npy", allow_pickle=True).item()
        dbs = pdh.carbachol_in_dark()
        base_folder = '/home/asyeda/pupil_analysis/dark_recordings'
        mname = 'LZ14' 
    else:
        raise ValueError("Invalid recording type. Only 'dark' is supported.")

    datexp = [db['datexp'] for db in dbs if db['mname'] == mname][0]
    _, _, _, tcams, tneurals, cumframes_neural, cumframes_cam, npz_data = nonlinear_refactored.load_data(base_folder, mname, datexp)

    # Load spks using the new deconvolution method (udcnv)
    spks_udcnv_file = "/media/Atika/neural_data_udcnv/{}_{}_1_spks.npz".format(mname, datexp)
    spks = np.load(spks_udcnv_file)['spks']
    frames_per_folder = npz_data['cam_frames_list']
    pupil_x = npz_data['pupil_x_polar']
    pupil_y = npz_data['pupil_y_polar']
    pupil_area = npz_data['pupil_area_small']

    # Load spks and pupil data
    n_folds = 5

    # Plot rastermap of before and after predictions
    pupil_x_all, pupil_y_all, pupil_area_all = [], [], []

    # Get raw data
    #y_test_all = rastermap_out[mname]['pupil']['spks']
    y_test_all = np.empty((rastermap_out[mname]['pupil']['U'].shape[0], 0))
    y_pred_pupil_all = np.empty((rastermap_out[mname]['pupil']['U'].shape[0], 0))
    y_pred_face_all = np.empty((rastermap_out[mname]['face_eye']['U'].shape[0], 0))
    for c_idx, condition in enumerate(['before', 'after']):
        y_test = np.empty((rastermap_out[mname]['pupil']['U'].shape[0], 0))
        y_pred_face = np.empty((rastermap_out[mname]['pupil']['U'].shape[0], 0))
        y_pred_pupil = np.empty((rastermap_out[mname]['pupil']['U'].shape[0], 0))
        for k_idx in range(n_folds):
            for segment in rastermap_out[mname]['pupil'][condition]['y_pred'][k_idx]:
                if segment.shape[0] > 0:
                    val = (segment @ rastermap_out[mname]['pupil']['U'].T)
                    y_pred_pupil = np.hstack((y_pred_pupil, val.T))

            for segment in rastermap_out[mname]['face_eye'][condition]['y_pred'][k_idx]:
                if segment.shape[0] > 0:
                    val = (segment @ rastermap_out[mname]['pupil']['U'].T)
                    y_pred_face = np.hstack((y_pred_face, val.T))

            # Get raw data
            for segment in rastermap_out[mname]['face_eye'][condition]['y_test'][k_idx]:
                if segment.shape[0] > 0:
                    val = (segment @ rastermap_out[mname]['pupil']['U'].T)
                    y_test = np.hstack((y_test, val.T))
        
        if condition == 'after':
            itest = rastermap_out[mname]['pupil'][condition]['itest']
            itest_indices = [elem for fold in itest for sub in fold for x in sub for elem in x]
            print("len itest_indices:", len(itest_indices), condition)
            y_pred_pupil_all =  np.hstack((y_pred_pupil_all, y_pred_pupil[:, np.argsort(itest_indices)]))
            blocks = list(npz_data['tcam'].item())
            blk = blocks[1].split('blk')[1]
            neural_timestamps = npz_data['tneural'].item()[f'blk{blk}'][np.sort(itest_indices)]
            camera_timestamps = npz_data['tcam'].item()[f'blk{blk}']    
            pupil_x_resampled = helper_functions.interp_value(helper_functions.segment_data(pupil_x, frames_per_folder, c_idx), camera_timestamps, neural_timestamps)
            pupil_y_resampled = helper_functions.interp_value(helper_functions.segment_data(pupil_y, frames_per_folder, c_idx), camera_timestamps, neural_timestamps)
            pupil_area_resampled = helper_functions.interp_value(helper_functions.segment_data(pupil_area, frames_per_folder, c_idx), camera_timestamps, neural_timestamps)
            #spks_blk = helper_functions.segment_data(spks.T, neural_frames_list, c_idx).T
            #y_test_all = np.hstack((y_test_all, spks_blk[:, np.sort(itest_indices)]))
            y_pred_face_all = np.hstack((y_pred_face_all, y_pred_face[:, np.argsort(itest_indices)]))
            y_test_all = np.hstack((y_test_all, y_test[:, np.argsort(itest_indices)]))
        else:
            itest = rastermap_out[mname]['face_eye'][condition]['itest']
            itest_indices = [elem for fold in itest for sub in fold for x in sub for elem in x]
            blk_end =  len(itest_indices)
            start = blk_end - np.ceil(1.5*60*3).astype(int)
            end = blk_end + (5*60*3)
            xlim_neural = [start,end]
            start_pupil = frames_per_folder[0] - np.ceil(1.5*60*3).astype(int)
            end_pupil = frames_per_folder[0] + (5*60*3)
            xlim_pupil = [start_pupil, end_pupil]
            pupil_x_resampled = np.full(blk_end, np.nan)
            pupil_y_resampled = np.full(blk_end, np.nan)
            neural_timestamps = npz_data['tneural'].item()['blk1'][np.sort(itest_indices)]
            camera_timestamps = npz_data['tcam'].item()['blk1']    
            pupil_area_resampled = helper_functions.interp_value(helper_functions.segment_data(pupil_area, frames_per_folder, c_idx), camera_timestamps, neural_timestamps)
            #spks_blk = helper_functions.segment_data(spks.T, neural_frames_list, c_idx).T
            #y_test_all = np.hstack((y_test_all, spks_blk[:, np.sort(itest_indices)]))
            y_pred_face_all =  np.hstack((y_pred_face_all, y_pred_face[:, np.argsort(itest_indices)]))
            y_test_all = np.hstack((y_test_all, y_test[:, np.argsort(itest_indices)]))

        pupil_x_all.append(pupil_x_resampled)
        pupil_y_all.append(pupil_y_resampled)
        pupil_area_all.append(pupil_area_resampled)

    pupil_x_all = np.concatenate(pupil_x_all, axis=0)
    pupil_y_all = np.concatenate(pupil_y_all, axis=0)
    pupil_area_all = np.concatenate(pupil_area_all, axis=0)

    saccade_data = np.load('/home/asyeda/pupil_analysis/saccade_times.npy', allow_pickle=True).item()
    aligned_saccades = []
    # Get saccades
    blocks = saccade_data["{}".format(recording_type)][mname].keys()
    for i, block in enumerate(blocks):
        _, a_sac = get_aligned_saccade_times(saccade_data["{}".format(recording_type)][mname][block], npz_data['tneural'].item()['blk{}'.format(block)])
        a_sac = a_sac.astype(int)
        if i==0:
            aligned_saccades.append(a_sac)
    aligned_saccades = np.hstack(aligned_saccades)

    # Run rastermap 
    # check if isort saved
    if os.path.exists(f"{base_folder}/{mname}_{datexp}_rastermap_isort.npy"):
        isort = np.load(f"{base_folder}/{mname}_{datexp}_rastermap_isort.npy")
    else:
        model = Rastermap().fit(zscore(spks, axis=1).copy())
        isort = model.isort
        np.save(f"{base_folder}/{mname}_{datexp}_rastermap_isort.npy", isort)

    # z-score the predicted data using mean and std of test data
    y_test_mean = np.nanmean(y_test_all, axis=1, keepdims=True)
    y_test_std = np.nanstd(y_test_all, axis=1, keepdims=True)
    y_pred_face_all = (y_pred_face_all - y_test_mean) / y_test_std
    y_pred_pupil_all = (y_pred_pupil_all - y_test_mean) / y_test_std
    y_test_all = (y_test_all - y_test_mean) / y_test_std

    # Sort the data based on the model's sorting and bin it
    y_test_embedding = utils.bin1d(y_test_all[isort], bin_size=200, axis=0)
    y_pred_face_embedding = utils.bin1d(y_pred_face_all[isort], bin_size=200, axis=0)
    y_pred_pupil_embedding = utils.bin1d(y_pred_pupil_all[isort], bin_size=200, axis=0)

    # Replace padding with NaN
    target_len = y_pred_face_embedding.shape[1]
    pupil_len = y_pred_pupil_embedding.shape[1]
    print("target_len:", target_len, "pupil_len:", pupil_len)
    # Create a canvas full of NaNs
    pupil_canvas = np.full((y_pred_pupil_embedding.shape[0], target_len), np.nan)
    # Put pupil data aligned at the end (so padding is at the start)
    pupil_canvas[:, -pupil_len:] = y_pred_pupil_embedding.copy()
    masked_pupil = pupil_canvas#np.ma.masked_invalid(pupil_canvas)

    # Colormap with transparent NaNs
    cmap = plt.cm.gray_r.copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    
    # Create the figure and GridSpec layout
    nested_gs = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs, hspace=.15, wspace=0.15)
    vmin, vmax = 0, 1.5

    # First subplot (Raw)
    ax1 = plt.subplot(nested_gs[0, 0])
    raster = ax1.imshow(y_test_embedding, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_title('Neural recording in the dark (test data)')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(xlim_neural)
    helper_functions.remove_spines(ax1)  
    # Add y-axis scale bar for rastermap
    bin_size = 200
    scale_bar_length = 10
    scale_bar_start = ax1.get_ylim()[0] - 10
    scale_bar_end = scale_bar_start + scale_bar_length
    # orient the text vertically
    ax1.plot([ax1.get_xlim()[0]-(0.01*np.diff(ax1.get_xlim())), ax1.get_xlim()[0]-(0.01*np.diff(ax1.get_xlim()))],
              [scale_bar_start, scale_bar_end], 'k-', lw=2, clip_on=False)
    ax1.text(ax1.get_xlim()[0]-(0.03*np.diff(ax1.get_xlim())), scale_bar_start-30, f'{scale_bar_length*bin_size} neurons',
              ha='left', va='center', rotation='vertical', fontsize=LABEL_FONTSIZE-3)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0], pos[1], pos[2], pos[3]*1.3])
    ax1.axvline(blk_end, color='r', linestyle='dashed')
    # set ylim
    ax1.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1])
    # Add x-axis scale bar for pupil area, x and y, rastermap on the right
    duration = 10 # seconds
    scale_bar_length = 3 * duration
    scale_bar_start = ax1.get_xlim()[1] - 70
    scale_bar_end = scale_bar_start + scale_bar_length
    y_pos = ax1.get_ylim()[0]+20
    ax1.plot([scale_bar_start, scale_bar_end], [y_pos, y_pos], 'k-', lw=1.5, clip_on=False)
    ax1.text(scale_bar_start + scale_bar_length/2, y_pos, f'{int(duration)} s'.format(str(duration)), 
             ha='center', va='bottom', clip_on=False, fontsize=LABEL_FONTSIZE-2)
    
    pos = ax1.get_position()
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
    cax.set_ylabel("z-scored", labelpad=-.4, fontsize=LABEL_FONTSIZE-3)  
    cax.set_yticks([vmin, vmax], labels=[f'{vmin}', f'{vmax}'], fontsize=LABEL_FONTSIZE-4)
                   
    # Second subplot (Predicted)
    ax2 = plt.subplot(nested_gs[1, 0])
    ax2.imshow(y_pred_face_embedding, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_title('Prediction from Face w/o eye', color='darkorange')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim(xlim_neural)
    helper_functions.remove_spines(ax2)  
    pos = ax2.get_position().bounds
    ax2.set_position([pos[0], pos[1]-.056, pos[2], pos[3]*1.3])
    ax2.axvline(blk_end, color='r', linestyle='dashed')

    # Third subplot (Pupil)
    ax3 = plt.subplot(nested_gs[2, 0])
    ax3.imshow(masked_pupil, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax3.set_title('Prediction from pupil position', color='darkgreen')
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_xlim(xlim_neural)
    ax3.axvline(blk_end, color='r', linestyle='dashed')
    helper_functions.remove_spines(ax3)
    # Add text "no prediction" in nan area
    ax3.text((xlim_neural[0]+blk_end)/2, ax3.get_ylim()[0]/2, 
             'No prediction', ha='center', va='center', fontsize=LABEL_FONTSIZE+2, 
             color='grey', rotation=45)
    pos = ax3.get_position().bounds
    ax3.set_position([pos[0], pos[1]-.115, pos[2], pos[3]*1.3])
    
    # Fourth subplot (Pupil X and Y)
    ax4 = plt.subplot(nested_gs[3, 0])
    ax4.plot(pupil_y_all, label='Elevation', lw=.5, color=pupil_y_color)
    ax4.plot(pupil_x_all, label='Azimuth', lw=.5, color=pupil_x_color)
    ax4.set_xlim(xlim_neural)
    ax4.set_xticks([])
    ax4.margins(x=0)   
    helper_functions.remove_spines(ax4)  
    ax4.set_yticks([])
    legend=ax4.legend(loc='center left', bbox_to_anchor=(-0.15, 0.45),facecolor='none', fontsize=LABEL_FONTSIZE-2)
    legend.get_texts()[0].set_color(pupil_y_color)  
    legend.get_texts()[1].set_color(pupil_x_color)  
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)
    #ax4.axvline(blk_end, color='r', linestyle='dashed')
    # Add y-axis scale bar for pupil x and y
    scale_bar_length = 10
    scale_bar_start = ax4.get_ylim()[0] + 25
    scale_bar_end = scale_bar_start + scale_bar_length
    #ax4.plot([blk_end-17, blk_end-17], [scale_bar_start, scale_bar_end], 'k-', lw=1, clip_on=False, zorder=10)
    #ax4.text(blk_end-65, scale_bar_start + scale_bar_length/2, f'{scale_bar_length}$^\circ$', ha='left', va='center', clip_on=False)
    pos = ax4.get_position().bounds
    ax4.set_position([pos[0], pos[1]-.01, pos[2], pos[3]*.3])
    
    # Fourth subplot (pupil area)
    ax5 = plt.subplot(nested_gs[4, 0])
    line, = ax5.plot(pupil_area_all, label='Pupil\narea', color='olive', lw=1.2)
    legend=ax5.legend(loc='center left', bbox_to_anchor=(-0.15, 0.55), facecolor='none', fontsize=LABEL_FONTSIZE-2)
    legend.get_texts()[0].set_color('olive') 
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)    
    ax5.set_xlim(xlim_neural)
    ax5.margins(x=0)
    helper_functions.remove_spines(ax5)
    ax5.set_yticks([])
    ax5.set_xticks([])
    #ax5.axvline(blk_end, color='r', linestyle='dashed')
    # adjust pos of ax5 
    pos = ax5.get_position().bounds
    ax5.set_position([pos[0], pos[1]+.085, pos[2], pos[3]*.3])
    
    """
    # Under vline add text annotation carbachol added
    #ax5.text(blk_end+10, ax5.get_ylim()[0], 'Carbachol added', ha='center', va='bottom', fontsize=10, color='r', clip_on=False)
    # Under vline add text annotation carbachol added
    # add saccade lines to all plots
    for s in aligned_saccades:
        ax1.axvline(s+blk_end, color=(0,1,0), linestyle='dashed', linewidth=vline)
        ax2.axvline(s+blk_end, color=(0,1,0), linestyle='dashed', linewidth=vline)
        ax3.axvline(s+blk_end, color=(0,1,0), linestyle='dashed', linewidth=vline)
        ax4.axvline(s+blk_end, color=(0,1,0), linestyle='dashed', linewidth=vline)
        ax5.axvline(s+blk_end, color=(0,1,0), linestyle='dashed', linewidth=vline)
    """

def zscore_neural_data(data, axis=1):
    data -= data.mean(axis=axis, keepdims=True)
    std = ((data ** 2).mean(axis=axis, keepdims=True) ** 0.5)
    std[std == 0] = 1
    data /= std
    return data