import matplotlib.pyplot as plt 
import os, sys
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import zscore
from rastermap import Rastermap, utils
from scipy.stats import f_oneway, ttest_rel
from statsmodels.stats.multitest import multipletests
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sys.path.insert(0, '/home/asyeda/Facemap/facemap')
from facemap import pupil
from facemap.pose import refine_pose
from itertools import combinations
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import cv2
from facemap import utils
from facemap.pose import refine_pose
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects

LABEL_FONTSIZE = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = LABEL_FONTSIZE
plt.rcParams['axes.titlesize'] = LABEL_FONTSIZE
plt.rcParams['axes.labelsize'] = LABEL_FONTSIZE-1
plt.rcParams['xtick.labelsize'] = LABEL_FONTSIZE-2
plt.rcParams['ytick.labelsize'] = LABEL_FONTSIZE-2
plt.rcParams['legend.fontsize'] = LABEL_FONTSIZE-2
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.titlelocation"] = "left"
plt.rcParams["axes.titleweight"] = "normal"

whisker_color = "#c92814ff"
nose_color = "#6bfbbaff"

# Set folder
BASE_FOLDER = 'set_your_base_folder_here/' 

def set_path(data_path):
    global BASE_FOLDER
    BASE_FOLDER = data_path

def plot(data_path):
    
    set_path(data_path)

    fig = plt.figure(figsize=(14, 7), dpi=300)
    gs = GridSpec(4, 6, figure=fig)

    plot_mouse_face_keypoints(gs[0, 0])
    plot_behavior_traces(gs[1, 0])
    plot_psth(gs[:2, 1:5])
    plot_saccade_frequency_during_whisking_sniffing(gs[:2, 5])

    plot_saccade_direction_relations(gs[2:, :3])
    plot_prediction_traces(gs[2:, 3:5])
    plot_correlation_matrix(gs[2:, 5:])

    # save figure
    plt.savefig(f'{BASE_FOLDER}figures/figure3.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_mouse_face_keypoints(gs):
    ax = plt.subplot(gs)
    keypoints_filepath = f'{BASE_FOLDER}data/mouse_data/TX123/2024_06_20/TX123_2024_06_20_2_full_FacemapPose.h5'
    video_path = f'{BASE_FOLDER}data/mouse_data/TX123/2024_06_20/TX123_2024_06_20_2_full.mp4' 

    # Load the keypoints
    keypoints_data = utils.load_keypoints(refine_pose.BODYPARTS, keypoints_filepath)
    pose_x_coord = keypoints_data[0, :, :]
    pose_y_coord = keypoints_data[1, :, :]
    colors = cm.get_cmap("jet")(np.linspace(0, 1.0, len(refine_pose.BODYPARTS)))

    # plot keypoints on frame
    frame_idx = 65332
    frame = get_frame_by_index(video_path, frame_idx)

    # Plot the image
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    alpha = 2.  # Contrast multiplier
    beta = 1     # Brightness offset (can be removed or adjusted)
    contrast_frame = cv2.convertScaleAbs(grayscale_frame, alpha=alpha, beta=beta)
    ax.imshow(contrast_frame, cmap='gray')

    # Add pupil circle on eye 
    proc_filepath = f"{BASE_FOLDER}data/mouse_data/TX123/2024_06_20/TX123_2024_06_20_1_full_proc.npy"
    proc_data = np.load(proc_filepath, allow_pickle=True).item()
    proc_frame_idx = frame_idx + 74081 # first block concatenated with second block
    frame = get_frame_by_index(video_path, frame_idx) 

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
    ax.scatter(xy[:, 1]+crop_x[0], xy[:, 0]+crop_y[0], c='green', s=.1, 
               marker='+', zorder=5, edgecolors='none')

    exclude_bp = ['paw', 'mouth', 'lowerlip']

    whisker_x_centroid, whisker_y_centroid = [], []
    nose_x_centroid, nose_y_centroid = [], []
    for bp in refine_pose.BODYPARTS:
        idx = refine_pose.BODYPARTS.index(bp)
        if bp not in exclude_bp and 'eye' not in bp and '(r)' not in bp:
            ax.scatter(pose_x_coord[idx, frame_idx], pose_y_coord[idx, frame_idx], c=colors[idx],
                        s=15, label=bp, alpha=0.5, zorder=5, edgecolors='none')
        if 'whisker' in bp:
            whisker_x_centroid.append(pose_x_coord[idx, frame_idx])
            whisker_y_centroid.append(pose_y_coord[idx, frame_idx])
        elif 'nose' in bp and 'bridge' not in bp:
            nose_x_centroid.append(pose_x_coord[idx, frame_idx])
            nose_y_centroid.append(pose_y_coord[idx, frame_idx])
    # Plot centroid
    ax.scatter(np.mean(whisker_x_centroid), np.mean(whisker_y_centroid), c='#c92814ff', s=50, label='Whiskers', marker='x')
    ax.scatter(np.mean(nose_x_centroid), np.mean(nose_y_centroid), c='#6bfbbaff', s=50, label='Nose', marker='x')
    ax.axis('off')
    ax.set_title("Face view")
    ax.set_xlim(30, 225)
    ax.set_ylim(210, 35)
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-.02, pos[1], pos[2]*1.2, pos[3]*1.2])

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

def plot_behavior_traces(gs):
    dbs = get_light_static_info() 
    mname = 'TX140'
    xlim = [118000, 119000]  # x-axis limits for the plot
    whisker_color = "#c92814ff"
    nose_color = "#6bfbbaff"
    # Load the data
    datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]
    # Load data
    _, _, _, _, _, _, _, npz_data = load_data(f"{BASE_FOLDER}data/npz_data/", mname, datexp)
            
    pupil_x = npz_data['pupil_x_small']
    pupil_y = npz_data['pupil_y_small']
    pose_x_coord = npz_data['pose_x_coord']
    pose_y_coord = npz_data['pose_y_coord']
    bodyparts = refine_pose.BODYPARTS
    num_classes = len(bodyparts)
    colors = cm.get_cmap("jet")(np.linspace(0, 1.0, num_classes))
    colors *= 255
    colors = colors.astype(int)
    lw = 1

    nested_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs)

    # Plot pupil x and y
    pupil_x_color = 'darkgreen'
    pupil_y_color = 'lightgreen'
    ax1 = plt.subplot(nested_gs[0])
    ax1.plot(pupil_x, color=pupil_x_color, label='Pupil X', lw=lw)
    ax1.plot(pupil_y, color=pupil_y_color, label='Pupil Y', lw=lw)
    ax1.set_xlim(xlim)
    ax1.text(.1, .73, 'Pupil y', transform=ax1.transAxes, ha='center', fontsize=LABEL_FONTSIZE-2, color="lightgreen")
    ax1.text(.1, .9, 'Pupil x', transform=ax1.transAxes, ha='center', fontsize=LABEL_FONTSIZE-2, color="darkgreen")
    ax1.set_yticks([])
    ax1.set_xticks([])
    remove_spines(ax1)

    lw = .6
    # Plot pose x and y for whiskers
    ax2 = plt.subplot(nested_gs[1])
    indices = [bodyparts.index(part) for part in bodyparts if 'whisker' in part]
    whisker_x_centroid = np.nanmean(pose_x_coord[indices, :], axis=0)
    whisker_x_centroid = whisker_x_centroid - whisker_x_centroid.mean()
    whisker_y_centroid = np.nanmean(pose_y_coord[indices, :], axis=0) 
    whisker_y_centroid = whisker_y_centroid - whisker_y_centroid.mean()
    ax2.plot(whisker_x_centroid, color='#f6471dff', label='Whiskers X', lw=lw)
    ax2.plot(whisker_y_centroid, color="#f6471dff", linestyle=':', label='Whiskers Y', lw=lw)
    ax2.set_xlim(xlim)
    ax2.set_yticks([])
    ax2.set_xticks([])
    remove_spines(ax2)
    ax2.text(0.15, 0.45, 'Whisker centroid', transform=ax2.transAxes, ha='center', fontsize=LABEL_FONTSIZE-2, color=whisker_color)
    h0,=ax2.plot(-100, 0,'k-', lw=lw)
    h1,=ax2.plot(-100, 0,'k--', lw=lw)
    ax2.legend([h0, h1], ['x keypoint', 'y keypoint'], loc=[-.2, .5], frameon=False,
               fontsize=LABEL_FONTSIZE-2, ncols=2, handlelength=1, handletextpad=.4)

    # Plot pose x and y for nose
    ax3 = plt.subplot(nested_gs[2])
    indices = [bodyparts.index(part) for part in bodyparts if 'nose' in part and 'bridge' not in part]
    nose_x_centroid = pose_x_coord[indices, :] #np.nanmean(pose_x_coord[indices, :], axis=0)
    nose_x_centroid = nose_x_centroid.T - nose_x_centroid.mean(axis=1)
    nose_y_centroid = pose_y_coord[indices, :] #np.nanmean(pose_y_coord[indices, :], axis=0)
    nose_y_centroid = nose_y_centroid.T - nose_y_centroid.mean(axis=1)
    ax3.plot(nose_x_centroid, color="#6bfbbaff", label='Nose X', lw=lw)
    ax3.plot(nose_y_centroid, color="#6bfbbaff", linestyle=':', label='Nose Y', lw=lw)
    ax3.set_xlim(xlim)
    ax3.set_yticks([])
    ax3.set_xticks([])
    remove_spines(ax3)
    ax3.text(0.15, .82, 'Nose centroid', transform=ax3.transAxes, ha='center', fontsize=LABEL_FONTSIZE-2, color=nose_color)

    # stretch plots vertically
    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width, pos.height*2])
    ax1.set_facecolor('none')
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0-.085, pos.width, pos.height*6])
    ax2.set_facecolor('none')
    pos = ax3.get_position()
    ax3.set_position([pos.x0, pos.y0-.2, pos.width, pos.height*6])
    ax3.set_facecolor('none')
    # move all plots lower
    for ax in [ax1, ax2, ax3]:
        pos = ax.get_position()
        ax.set_position([pos.x0-.01, pos.y0-.03, pos.width, pos.height])

def remove_spines(ax):
    for loc, spine in ax.spines.items():
        spine.set_linewidth(0)

def plot_saccade_frequency_during_whisking_sniffing(gs):

    if os.path.exists(f'{BASE_FOLDER}data/saccade_data/saccade_orofacial_freq.npy'):
        freq_data = np.load(f'{BASE_FOLDER}data/saccade_data/saccade_orofacial_freq.npy', allow_pickle=True).item()
        whisking_saccade_freq = freq_data['whisking_saccade_freq']
        sniffing_saccade_freq = freq_data['sniffing_saccade_freq']
        saccade_freq = freq_data['saccade_freq']
    else:
        # Load databases
        dark_dbs = carbachol_in_dark() 
        light_random_dbs = get_light_random_info()
        light_static_dbs = get_light_static_info()
        base_folders = [
            f'{BASE_FOLDER}dark_recordings',
            f'{BASE_FOLDER}light_recordings_random',
            f'{BASE_FOLDER}light_recordings_static']
        dbs = [dark_dbs, light_random_dbs, light_static_dbs]
        category = ['dark', 'light_random', 'light_static']
        saccade_data = np.load(f'{BASE_FOLDER}data/saccade_data/saccade_times.npy', allow_pickle=True).item()

        BODYPARTS = [
            "eye(back)",
            "eye(bottom)",
            "eye(front)",
            "eye(top)",
            "lowerlip",
            "mouth",
            "nose(bottom)",
            "nose(r)",
            "nose(tip)",
            "nose(top)",
            "nosebridge",
            "paw",
            "whisker(I)",  
            "whisker(III)",  
            "whisker(II)",  
        ]

        whisking_saccade_freq = []
        sniffing_saccade_freq = []
        saccade_freq = []

        for db_idx, db in enumerate(dbs):
            # Get mouse list
            mouse_list = np.unique([entry['mname'] for entry in db])
            for mname in mouse_list:
                print('Processing mouse: ', mname)
                # Get datexp 
                datexp = [entry['datexp'] for entry in db if entry['mname'] == mname][0]
                # Load data
                _, _, _, _, _, _, _, npz_data = load_data(f'{BASE_FOLDER}data/npz_data/', mname, datexp)

                # segment the data into blocks
                block_frames = npz_data['cam_frames_list']
                
                # Get number of blocks in the recording
                blocks = list(npz_data['tcam'].item())
                num_blocks = len(blocks)

                if 'light' in category[db_idx]:
                    block_idxs = np.arange(1)
                else:
                    block_idxs = np.arange(1, num_blocks)

                for blk_idx in block_idxs: 
                    blk = blocks[blk_idx].split('blk')[1]
                    if category[db_idx] == 'dark' and blk == '1':
                        continue
                    print('Processing block: ', blk)
                    # Get pupil timestamp
                    pupil_timestamp = npz_data['tcam'].item()[list(npz_data['tcam'].item())[blk_idx]]
                    pupil_x = segment_data(npz_data["pupil_x_small"], block_frames, blk_idx)
                    pupil_y = segment_data(npz_data["pupil_y_small"] , block_frames, blk_idx)
                    pupil_area = segment_data(npz_data["pupil_area_small"], block_frames, blk_idx)
                    pose_x_coord = segment_data(npz_data['pose_x_coord'].T, block_frames, blk_idx)
                    pose_y_coord = segment_data(npz_data['pose_y_coord'].T, block_frames, blk_idx)
                    blink = segment_data(npz_data["blink_small"], block_frames, blk_idx)
                    pupil_xy = np.vstack((pupil_x, pupil_y)).T

                    # replace nan with interpolated values
                    nans, x= np.isnan(pupil_area), lambda z: z.nonzero()[0]
                    pupil_area[nans]= np.interp(x(nans), x(~nans), pupil_area[~nans])
                    
                    # zscore pupil area
                    pupil_area = (pupil_area - np.nanmean(pupil_area)) / np.nanstd(pupil_area)

                    exact_saccades = saccade_data[category[db_idx]][mname][blk] # Get exact saccade indices from precomputed data

                    # Extract pose values before and after the saccade
                    pose_whisker_x = pose_x_coord[:, -3:]  # Last 3 keypoints for whiskers
                    pose_whisker_y = pose_y_coord[:, -3:]  # Last 3 keypoints for whiskers
                    pose_nose_x = pose_x_coord[:, 6:10]  # Nose keypoints excluding nosebridge
                    pose_nose_y = pose_y_coord[:, 6:10]  # Nose keypoints excluding nosebridge

                    # Get centroid of whisker, nose, and eye keypoints
                    whisker_x_centroid = np.mean(pose_whisker_x, axis=1)
                    whisker_y_centroid = np.mean(pose_whisker_y, axis=1)
                    whisker_centroid_dist = np.sqrt(np.diff(whisker_x_centroid)**2 + np.diff(whisker_y_centroid)**2) # np.diff(whisker_y_centroid) #

                    nose_x_centroid = np.mean(pose_nose_x, axis=1)
                    nose_y_centroid = np.mean(pose_nose_y, axis=1)
                    nose_centroid_dist = np.sqrt(np.diff(nose_x_centroid)**2 + np.diff(nose_y_centroid)**2) #np.diff(nose_y_centroid) #

                    # saccade
                    exact_saccades = saccade_data[category[db_idx]][mname][blk] # Get exact saccade indices from precomputed data
                    sac_start_idx = np.array(exact_saccades)[:,0].astype(int)
                    sac_end_idx = np.array(exact_saccades)[:,1].astype(int)
                    # get binary array of saccade times
                    saccade_times = np.zeros_like(pupil_x)
                    for start, end in zip(sac_start_idx, sac_end_idx):
                        saccade_times[start:end] = 1

                    percentile = 97 

                    threshold = np.percentile(whisker_centroid_dist, percentile)
                    # Create a binary array indicating whisker_centroid_dist > threshold
                    whisker_movement_times = (whisker_centroid_dist > threshold).astype(int)
                    # Find overlapping times
                    overlapping_times = np.where((saccade_times[1:] == 1) & (whisker_movement_times == 1))[0]
                    print("Number of overlapping times: ", len(overlapping_times))

                    threshold   = np.percentile(nose_centroid_dist, percentile)
                    nose_movement_times = (nose_centroid_dist > threshold).astype(int)
                    # Find overlapping times
                    overlapping_times_nose = np.where((saccade_times[1:] == 1) & (nose_movement_times == 1))[0]
                    print("Number of overlapping times nose: ", len(overlapping_times_nose))
                    # Calculate frequency of saccade during nose movement
                    if len(np.where(nose_movement_times == 1)[0]) > 0:
                        saccade_during_nose_movement_freq = len(overlapping_times_nose) / np.sum(saccade_times == 1)
                                                                                                    #len(np.where(nose_movement_times == 1)[0])
                    else:
                        saccade_during_nose_movement_freq = 0

                    # Calculate frequency of saccade during whisker movement
                    if len(np.where(whisker_movement_times == 1)[0]) > 0:
                        saccade_during_whisker_movement_freq = len(overlapping_times) / np.sum(saccade_times == 1)#len(np.where(whisker_movement_times == 1)[0])
                    else:
                        saccade_during_whisker_movement_freq = 0

                    # Append to the list
                    whisking_saccade_freq.append(saccade_during_whisker_movement_freq)
                    sniffing_saccade_freq.append(saccade_during_nose_movement_freq)
                    saccade_freq.append(np.sum(saccade_times == 1) / (len(saccade_times))) # saccades per second
        # Save the computed frequencies
        freq_data = {
            'whisking_saccade_freq': whisking_saccade_freq,
            'sniffing_saccade_freq': sniffing_saccade_freq,
            'saccade_freq': saccade_freq
        }
        np.save(f'{BASE_FOLDER}data/saccade_data/saccade_orofacial_freq.npy', freq_data)

    ax = plt.subplot(gs)
    # Plot data points
    x_positions = [0, 1, 2]
    data_groups = [whisking_saccade_freq, sniffing_saccade_freq, saccade_freq]
    colors = [whisker_color, nose_color, 'grey']
    labels = ['Whisking', 'Sniffing', 'All']

    # print mean +- s.d. for each group
    for i, data in enumerate(data_groups):
        mean = np.mean(data)
        sd = np.std(data)
        print(f"{labels[i]}: {mean:.3f}$\pm${sd:.3f}")

    for i, (x, data, color) in enumerate(zip(x_positions, data_groups, colors)):
        ax.plot(np.ones(len(data)) * x, data, 'o', color=color, alpha=0.7, markersize=3)

    # Connect paired points
    for i in range(len(whisking_saccade_freq)):
        ax.plot(x_positions, [whisking_saccade_freq[i],
                            sniffing_saccade_freq[i],
                            saccade_freq[i]], color='grey', lw=0.5, alpha=0.5)

    ax.set_xticks(x_positions)
    ax.set_xlim(-0.25, 2.25)
    ax.set_xticklabels(labels, fontsize=LABEL_FONTSIZE, rotation=90)
    ax.set_ylabel('Saccade probability', fontsize=LABEL_FONTSIZE, labelpad=-5)
    ax.set_yticks([0, 0.3])
    ax.set_yticklabels([0, 0.3], fontsize=LABEL_FONTSIZE-2)

    # Adjust layout
    pos = ax.get_position()
    ax.set_position([pos.x0+0.1, pos.y0+.06, pos.width*.5, pos.height*.8])

    # Create inset axis above for significance bars
    axin = ax.inset_axes([0.0, 1.01, 1, 0.1])  # x, y, width, height (relative to ax)
    axin.axis('off')
    # Perform all pairwise t-tests
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_labels = [('Whisking', 'Sniffing'),
                ('Whisking', 'All'),
                ('Sniffing', 'All')]
    y_offset = 0.05   # distance between stacked bars
    bar_height = 0.05 # relative to y-axis range

    for i, (a, b) in enumerate(pairs):
        stat, p_value = ttest_rel(data_groups[a], data_groups[b])
        # Base y-position for this bar (stacked)
        y = y_offset * (i + 1)
        # Call with *fixed* height factor
        add_significance_bar(axin, a, b, y=y, p=p_value,
                            color='k', text_offset=-0.04)


def plot_keypoints_net(gs, filepath):
    # Load the image
    img = plt.imread(filepath)

    # Create inset axes
    ax = plt.subplot(gs)
    ax.imshow(img)
    ax.axis('off')  # Hide the axes
    pos = ax.get_position()
    ax.set_position([pos.x0-.15, pos.y0+.04, pos.width*1.8, pos.height*1.4])

def plot_correlation_matrix(gs):

    # Load data
    data_dir = f'{BASE_FOLDER}data/saccade_data/bodypart_group_results_zscore.npz'
    results = np.load(data_dir, allow_pickle=True)
    results_df = pd.DataFrame({key: results[key] for key in results.keys()})

    # Exclude 'eye' group if it exists
    results_df = results_df[
        (results_df['input_group'] != 'eye') & (results_df['output_group'] != 'eye')
    ]

    # Define desired group order
    group_order = ['pupil', 'nose', 'whisker']

def plot_correlation_matrix(gs):

    # Load data
    data_dir = f'{BASE_FOLDER}data/saccade_data/bodypart_group_results_zscore.npz'
    results = np.load(data_dir, allow_pickle=True)
    results_df = pd.DataFrame({key: results[key] for key in results.keys()})

    # Exclude 'eye' group if it exists
    results_df = results_df[
        (results_df['input_group'] != 'eye') & (results_df['output_group'] != 'eye')
    ]

    # Define desired group order
    group_order = ['pupil', 'nose', 'whisker']

    # Create nested gridspec for two vertically stacked heatmaps
    nested_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs, hspace=0.4, height_ratios=[1, 1]
    )

    vmin, vmax = 0, 1

    def get_mean_sem(metric_key):
        mean_df = (
            results_df.groupby(['output_group', 'input_group'])[metric_key]
            .mean()
            .unstack()
            .reindex(index=group_order, columns=group_order)
        )
        sem_df = (
            results_df.groupby(['output_group', 'input_group'])[metric_key]
            .sem()
            .unstack()
            .reindex(index=group_order, columns=group_order)
        )
        annot_df = mean_df.round(2).astype(str) + "\n± " + sem_df.round(2).astype(str)
        annot_df = annot_df.where(mean_df.notna(), "")
        return mean_df, annot_df

    # Create the figure-level colorbar axes
    ax1 = plt.subplot(nested_gs[0])
    ax2 = plt.subplot(nested_gs[1])

    # Plot both heatmaps without individual colorbars
    mean_x, annot_x = get_mean_sem('varexp_x')
    mean_y, annot_y = get_mean_sem('varexp_y')

    hm1 = sns.heatmap(
        mean_x, ax=ax1, annot=annot_x, fmt="", cmap="viridis",
        vmin=vmin, vmax=vmax, cbar=False, annot_kws={"size": 10}
    )
    hm2 = sns.heatmap(
        mean_y, ax=ax2, annot=annot_y, fmt="", cmap="viridis",
        vmin=vmin, vmax=vmax, cbar=False, annot_kws={"size": 10}
    )

    # Adjust labels and ticks
    for ax, coord in zip([ax1, ax2], ['x', 'y']):
        yticklabels = [f"{g} ({coord})" for g in group_order]
        ax.set_yticklabels(yticklabels, rotation=0, fontsize=LABEL_FONTSIZE - 2)
        ax.set_ylabel("Predicted", fontsize=LABEL_FONTSIZE - 1)
        ax.tick_params(labelsize=LABEL_FONTSIZE - 2)
    ax1.set_xticklabels([])  # hide top x labels
    ax1.set_xlabel("")
    ax2.set_xlabel("Input group", fontsize=LABEL_FONTSIZE - 1)
    # adjust position of ax1 and ax2
    x_adjust = 0.022
    y_adjust = -0.135
    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0+x_adjust, pos1.y0+y_adjust, pos1.width, pos1.height*1.4])
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0+x_adjust, pos2.y0+y_adjust-.03, pos2.width, pos2.height*1.4])

    # Add ONE shared colorbar for both heatmaps
    fig = plt.gcf()
    im = ax2.collections[0]  # reference to colormap
    cbar_ax = fig.add_axes([0.93, 0.1, 0.005, 0.2])  # [x, y, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Variance explained", fontsize=LABEL_FONTSIZE - 2)
    cbar.ax.tick_params(labelsize=LABEL_FONTSIZE - 3)
    cbar.set_ticks([0, 0.5, 1])


def plot_saccade_durations(gs):
    # Load data
    data_dir = f'{BASE_FOLDER}data/saccade_data/saccade_durations.npz'
    saccade_data = np.load(data_dir, allow_pickle=True)
    saccade_durations = saccade_data['saccade_durations']

    # Plot saccade durations
    ax = plt.subplot(gs[0, 1])
    ax.hist(saccade_durations, bins=30, color='grey', edgecolor='k', alpha=0.7)
    ax.set_xlabel('Saccade Duration (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title("Saccade Duration Distribution")
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*1.1, pos.height])

def plot_psth(gs):
    # Load data
    keypoints_psth_path = f'{BASE_FOLDER}data/saccade_data/psth_whisker_nose_eye.npz'
    keypoints_psth_data = np.load(keypoints_psth_path, allow_pickle=True)
    mean_psth_whisker_all=keypoints_psth_data['mean_psth_whisker_all']
    mean_psth_nose_all=keypoints_psth_data['mean_psth_nose_all']
    mean_psth_eye_all=keypoints_psth_data['mean_psth_eye_all']
    mean_psth_pupilarea_all=keypoints_psth_data['mean_psth_pupilarea_all']

    mean_psth_whisker_random_all=keypoints_psth_data['mean_psth_whisker_random_all']
    mean_psth_nose_random_all=keypoints_psth_data['mean_psth_nose_random_all']
    mean_psth_eye_random_all=keypoints_psth_data['mean_psth_eye_random_all']
    mean_psth_pupilarea_random_all=keypoints_psth_data['mean_psth_pupilarea_random_all']

    sem_psth_whisker_all=keypoints_psth_data['sem_psth_whisker_all']
    sem_psth_nose_all=keypoints_psth_data['sem_psth_nose_all']
    sem_psth_eye_all=keypoints_psth_data['sem_psth_eye_all']
    sem_psth_pupilarea_all=keypoints_psth_data['sem_psth_pupilarea_all']

    sem_psth_whisker_random_all=keypoints_psth_data['sem_psth_whisker_random_all']
    sem_psth_nose_random_all=keypoints_psth_data['sem_psth_nose_random_all']
    sem_psth_eye_random_all=keypoints_psth_data['sem_psth_eye_random_all']
    sem_psth_pupilarea_random_all=keypoints_psth_data['sem_psth_pupilarea_random_all']

    post_avg_whisker=keypoints_psth_data['post_avg_whisker']
    post_avg_nose=keypoints_psth_data['post_avg_nose']
    post_avg_pupilarea=keypoints_psth_data['post_avg_pupilarea']
    pre_avg_whisker=keypoints_psth_data['pre_avg_whisker']
    pre_avg_nose=keypoints_psth_data['pre_avg_nose']
    pre_avg_pupilarea=keypoints_psth_data['pre_avg_pupilarea']

    post_avg_random_whisker=keypoints_psth_data['post_avg_random_whisker']
    post_avg_random_nose=keypoints_psth_data['post_avg_random_nose']
    random_avg_pupilarea=keypoints_psth_data['random_avg_pupilarea']
    sampling_rate=keypoints_psth_data['sampling_rate']
    padding=keypoints_psth_data['padding']

    # Plot PSTH 
    nested_gs = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs, wspace=.8)

    # ~~~~~~~~~~~ Whisker and nose speed ~~~~~~~~~~~
    # Plot mean values with shaded error
    line_lw = 1
    ax0 = plt.subplot(nested_gs[0, 0])
    ax0.plot(mean_psth_whisker_all, label="Whiskers", lw=line_lw, color=whisker_color)
    ax0.fill_between(np.arange(len(mean_psth_whisker_all)), mean_psth_whisker_all-sem_psth_whisker_all, mean_psth_whisker_all+sem_psth_whisker_all, alpha=0.2, color=whisker_color)
    ax0.plot(mean_psth_nose_all, label="Nose", lw=line_lw, color=nose_color)
    ax0.fill_between(np.arange(len(mean_psth_nose_all)), mean_psth_nose_all-sem_psth_nose_all, mean_psth_nose_all+sem_psth_nose_all, alpha=0.2, color=nose_color)
    # Plot random mean values
    ax1 = plt.subplot(nested_gs[0, 1])
    ax1.plot(mean_psth_whisker_random_all, label="Whiskers", lw=line_lw, color=whisker_color)
    ax1.fill_between(np.arange(len(mean_psth_whisker_random_all)), mean_psth_whisker_random_all-sem_psth_whisker_random_all, mean_psth_whisker_random_all+sem_psth_whisker_random_all, alpha=0.2, color=whisker_color)
    ax1.plot(mean_psth_nose_random_all, label="Nose", lw=line_lw, color=nose_color)
    ax1.fill_between(np.arange(len(mean_psth_nose_random_all)), mean_psth_nose_random_all-sem_psth_nose_random_all, mean_psth_nose_random_all+sem_psth_nose_random_all, alpha=0.2, color=nose_color)
    # Add legend
    leg = ax0.legend(loc='lower left', frameon=False)
    leg_colors = [whisker_color, nose_color]
    for l in leg.get_texts():
        l.set_color(leg_colors.pop(0))
        l.set_path_effects([patheffects.withStroke(linewidth=.2, foreground='k')])
    for leg_line in leg.get_lines():
        leg_line.set_linewidth(0)
    # Calculate the center point (where vline should be at 0)
    center_point = padding
    # Assume mean_psth_whisker_all is your data, center_point is index of saccade start
    n_samples = len(mean_psth_whisker_all)
    time_vector = (np.arange(n_samples) - center_point) / sampling_rate  # in seconds
    # Now choose ticks based on time
    tick_spacing_sec = 1  # for example, ticks every 0.2 seconds
    min_time = time_vector[0]
    max_time = time_vector[-1]
    xtick_labels = np.arange(np.ceil(min_time / tick_spacing_sec) * tick_spacing_sec,
                            np.floor(max_time / tick_spacing_sec) * tick_spacing_sec + tick_spacing_sec +1,
                            tick_spacing_sec)
    # Map these labels back to data indices
    xticks = (xtick_labels * sampling_rate + center_point)
    xtick_labels = np.round(xtick_labels, 0).astype(int)
    
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xtick_labels)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)

    # Apply the ticks and labels to both subplots
    # set same ylim for both subplots
    max_ylim = np.max([ax0.get_ylim()[1], ax1.get_ylim()[1]])
    ax0.set_ylim(0, max_ylim)
    ax1.set_ylim(0, max_ylim)
    for a in [ax0, ax1]:
        a.set_xlim(0, len(mean_psth_whisker_all))
        # Add vertical line at center point (which will now correspond to 0 in the labels)
        ymin, ymax = np.min([ax0.get_ylim()[0], ax1.get_ylim()[0]]), np.max([ax0.get_ylim()[1], ax1.get_ylim()[1]])
        a.vlines(center_point, ymin=0, ymax=ymax, color='k', linestyle='--', lw=.8)
    ax0.set_ylabel('Speed (px/s)')

    # Remove spines
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    # show only 3 yticklabels
    ax0.set_yticks([0, 1, 2])
    ax0.set_yticklabels([0, 1, 2], fontsize=LABEL_FONTSIZE-2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_yticks([])

    # Plot stats test
    ax4 = plt.subplot(nested_gs[0, 2])

    # Make a plot saving the post and pre avg. plot as line plot showing each mouse
    ax4.plot([post_avg_random_whisker, pre_avg_whisker, post_avg_whisker], color=whisker_color, alpha=0.25, lw=0.5)
    ax4.plot([post_avg_random_nose, pre_avg_nose, post_avg_nose], color=nose_color, alpha=0.25, lw=0.5)
    
    # plot avg
    ax4.plot([np.mean(post_avg_random_whisker), np.mean(pre_avg_whisker), np.mean(post_avg_whisker)], 
            color=whisker_color, label='Whiskers', lw=3)
    ax4.plot([np.mean(post_avg_random_nose), np.mean(pre_avg_nose), np.mean(post_avg_nose)], 
            color=nose_color, label='Nose', lw=3)    
    ax4.set_xticks([0, 1, 2])
    ax4.set_ylim(0, 3.5)
    ax4.set_xticklabels(['Random', 'Pre-saccade', 'Post-saccade'], rotation=90)
    ax4.set_ylabel('Avg. speed (px/s)')
    # set 3 yticks only
    ax4.set_yticks([0, 1.5, 3])
    ax4.set_yticklabels([0, 1.5, 3], fontsize=LABEL_FONTSIZE-2)
    # Add legend
    leg = ax4.legend(loc='upper left', frameon=False, bbox_to_anchor=(-.4, 1))
    # remove line and color text same as linecolor
    leg_colors = [whisker_color, nose_color]
    for l in leg.get_texts():
        l.set_color(leg_colors.pop(0))
        l.set_path_effects([patheffects.withStroke(linewidth=.2, foreground='k')])
    for leg_line in leg.get_lines():
        leg_line.set_linewidth(0)
    # Add inset axis for significance annotations above the plot
    axin = ax4.inset_axes([0, 1.03, 1, 0.15])  # Position the inset axis above the main axis
    axin.axis('off')  # Hide the inset axis

    # Perform all pairwise comparisons for whiskers
    conditions = ['Random', 'Pre', 'Post']
    whisker_data = [post_avg_random_whisker, pre_avg_whisker, post_avg_whisker]
    nose_data = [post_avg_random_nose, pre_avg_nose, post_avg_nose]
    pupilarea_data = [random_avg_pupilarea, pre_avg_pupilarea, post_avg_pupilarea]

    # Parameters
    bar_height = 0.05  # fixed height for all significance bars
    whisker_spacing = 0.19  # vertical spacing between whisker bars
    nose_spacing = 0.19     # vertical spacing between nose bars
    y_base = max(np.mean(whisker_data), np.mean(whisker_data))

    # --- Whisker comparisons ---
    whisker_pairs = list(combinations([0, 1, 2], 2))
    for n, (i, j) in enumerate(whisker_pairs):
        data1, data2 = whisker_data[i], whisker_data[j]
        stat, p = ttest_rel(data1, data2) if len(data1) == len(data2) else (np.nan, np.nan)

        if not np.isnan(p) and p < 0.05:
            # y position stacked above the bars
            y_pos = y_base + whisker_spacing 
            add_significance_bar(axin, i, j, y_pos, p, height=bar_height, color=whisker_color, text_offset=-0.19)
            y_base = y_pos  # update base for next bar

    # --- Nose comparisons ---
    nose_pairs = list(combinations([0, 1, 2], 2))
    y_base = min(np.mean(nose_data), np.mean(nose_data))
    for n, (i, j) in enumerate(nose_pairs):
        data1, data2 = nose_data[i], nose_data[j]
        stat, p = ttest_rel(data1, data2) if len(data1) == len(data2) else (np.nan, np.nan)

        if not np.isnan(p) and p < 0.05:
            # y position stacked below the bars
            y_pos = y_base + nose_spacing #* (n + 1)
            add_significance_bar(axin, i, j, y_pos, p, height=bar_height, color=nose_color, text_offset=-0.21)
            y_base = y_pos  # update base for next bar

    # Load running psth data
    running_psth_path = f'{BASE_FOLDER}data/saccade_data/psth_running.npz'
    running_psth_data = np.load(running_psth_path, allow_pickle=True)
    random_avg_running = running_psth_data['random_avg_running']/10
    pre_avg_running = running_psth_data['pre_avg_running']/10
    post_avg_running = running_psth_data['post_avg_running']/10
    mean_psth_running_all = running_psth_data['mean_psth_running_speed_all']
    sem_psth_running_all = running_psth_data['sem_psth_running_speed_all']
    mean_psth_running_random_all = running_psth_data['mean_psth_running_speed_random_all']
    sem_psth_running_random_all = running_psth_data['sem_psth_running_speed_random_all']
    padding = running_psth_data['padding']
    sampling_rate = running_psth_data['sampling_rate']

    pupil_ylim = (-0.2, 1.2)
    running_ylim = (-1.4, 8.5)

    # Plot pupil area PSTH
    # --- Combined Pupil Area + Running (Saccade-aligned) ---
    ax2a = plt.subplot(nested_gs[0, 3])
    ax2 = ax2a.twinx()  # create twin y-axis

    # --- Pupil Area (left y-axis) ---
    ax2a.plot(mean_psth_pupilarea_all, label="Pupil area", lw=line_lw, color='olive')
    ax2a.fill_between(np.arange(len(mean_psth_pupilarea_all)),
                    mean_psth_pupilarea_all - sem_psth_pupilarea_all,
                    mean_psth_pupilarea_all + sem_psth_pupilarea_all,
                    alpha=0.2, color='olive')
    ax2a.hlines(0, xmin=0, xmax=len(mean_psth_pupilarea_all), color='k', linestyle='--', lw=.5)
    ax2a.set_ylim(pupil_ylim)
    ax2a.set_ylabel("Pupil area (z-scored)", color='olive')
    ax2a.tick_params(axis='y', colors='olive')
    ax2a.spines['top'].set_visible(False)

    # --- Running (right y-axis) ---
    ax2.plot(mean_psth_running_all / 10, label="Running", lw=line_lw, color='blue')
    ax2.fill_between(np.arange(len(mean_psth_running_all)),
                    mean_psth_running_all / 10 - sem_psth_running_all / 10,
                    mean_psth_running_all / 10 + sem_psth_running_all / 10,
                    alpha=0.2, color='blue')
    ax2.set_ylim(running_ylim)
    ax2.tick_params(axis='y', colors='blue')
    ax2.spines['top'].set_visible(False)

    # --- Time axis setup ---
    n_samples = len(mean_psth_pupilarea_all)
    time_vector = (np.arange(n_samples) - padding) / sampling_rate
    tick_spacing_sec = 1
    xtick_labels = np.arange(np.ceil(time_vector[0] / tick_spacing_sec) * tick_spacing_sec,
                            np.floor(time_vector[-1] / tick_spacing_sec) * tick_spacing_sec + tick_spacing_sec,
                            tick_spacing_sec)
    xticks = (xtick_labels * sampling_rate + padding)
    xtick_labels = np.round(xtick_labels, 1).astype(int)
    ax2a.set_xticks(xticks, xtick_labels)
    ax2a.spines['right'].set_visible(False)
    ax2a.set_yticks([0, 1])
    ax2a.set_yticklabels([0, 1], fontsize=LABEL_FONTSIZE-2)

    # vertical line
    #ax2a.vlines(padding, ymin=-0.2, ymax=1, color='k', linestyle='--', lw=.8)
    ax2.vlines(padding, ymin=running_ylim[0], ymax=running_ylim[1], color='k', linestyle='--', lw=.8)
    ax2.set_yticks([])

    # --- Combined Pupil Area + Running (Random-aligned) ---
    ax3a = plt.subplot(nested_gs[0, 4])
    ax3 = ax3a.twinx()

    # --- Pupil Area ---
    ax3a.plot(mean_psth_pupilarea_random_all, label="Pupil area", lw=line_lw, color='olive')
    ax3a.fill_between(np.arange(len(mean_psth_pupilarea_random_all)),
                    mean_psth_pupilarea_random_all - sem_psth_pupilarea_random_all,
                    mean_psth_pupilarea_random_all + sem_psth_pupilarea_random_all,
                    alpha=0.2, color='olive')
    ax3a.hlines(0, xmin=0, xmax=len(mean_psth_pupilarea_random_all), color='k', linestyle='--', lw=.5)
    ax3a.set_ylim(pupil_ylim)
    ax3a.tick_params(axis='y', colors='olive')
    ax3a.spines['top'].set_visible(False)
    ax3a.set_yticks([])
    ax3a.spines['left'].set_visible(False)

    # --- Running ---
    ax3.plot(mean_psth_running_random_all / 10, lw=line_lw, color='blue')
    ax3.fill_between(np.arange(len(mean_psth_running_random_all)),
                    mean_psth_running_random_all / 10 - sem_psth_running_random_all / 10,
                    mean_psth_running_random_all / 10 + sem_psth_running_random_all / 10,
                    alpha=0.2, color='blue')
    ax3.set_ylim(running_ylim)
    ax3.set_ylabel("Running speed (cm/s)", color='blue')
    ax3.tick_params(axis='y', colors='blue')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(True)
    ax3.spines['left'].set_visible(False)
    ax3.set_yticks([0, 4, 8])
    ax3.set_yticklabels([0, 4, 8], fontsize=LABEL_FONTSIZE-2)

    # --- Time axis ---
    n_samples = len(mean_psth_pupilarea_random_all)
    time_vector = (np.arange(n_samples) - padding) / sampling_rate
    xtick_labels = np.arange(np.ceil(time_vector[0] / tick_spacing_sec) * tick_spacing_sec,
                            np.floor(time_vector[-1] / tick_spacing_sec) * tick_spacing_sec + tick_spacing_sec,
                            tick_spacing_sec)
    xticks = (xtick_labels * sampling_rate + padding)
    xtick_labels = np.round(xtick_labels, 1).astype(int)
    ax3a.set_xticks(xticks, xtick_labels)
    ax3a.set_xlabel("Time from random (s)")

    # vertical line
    #ax3a.vlines(padding, ymin=-0.2, ymax=1, color='k', linestyle='--', lw=.8)
    ax3.vlines(padding, ymin=running_ylim[0], ymax=running_ylim[1], color='k', linestyle='--', lw=.8)

    ax5 = plt.subplot(nested_gs[0, 5])
    ax5.plot([random_avg_running, pre_avg_running, post_avg_running], alpha=0.25, lw=.5, color='blue')
    # plot avg
    ax5.plot([np.mean(random_avg_running), np.mean(pre_avg_running), np.mean(post_avg_running)], 
                color='blue', label='Running', lw=3)
    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(['Non-saccade', 'Pre-saccade', 'Post-saccade'], rotation=90)
    ax5.set_ylabel('Avg. speed (cm/s)', color='blue')
    ax5.tick_params(axis='y', colors='blue')
    ax5.set_ylim(running_ylim)
    ax5.set_yticks([0, 4, 8])
    ax5.set_yticklabels([0, 4, 8], fontsize=LABEL_FONTSIZE-2)
    ax5.spines['top'].set_visible(False)
    
    # Perform all pairwise comparisons for whiskers
    conditions = ['Random', 'Pre', 'Post']
    running_data = [random_avg_running, pre_avg_running, post_avg_running]

    axin = ax5.inset_axes([0, 1.02, 1, 0.15])  # Position the inset axis above the main axis
    axin.axis('off')  # Hide the inset axis

    # Parameters
    bar_height = 0.025  # fixed height for all significance bars
    running_spacing = 0.1  # vertical spacing between whisker bars
    y_base = max(np.mean(running_data), np.mean(running_data))

    for (i, j) in combinations([0, 1, 2], 2):
        data1, data2 = running_data[i], running_data[j]
        _, p = ttest_rel(data1, data2) if len(data1) == len(data2) else (np.nan, np.nan)        
        # Plot significance
        if not np.isnan(p) and p < 0.05:
            y_pos = y_base + running_spacing 
            add_significance_bar(axin, i, j, y_pos, p, height=bar_height, color='blue', text_offset=-0.03)
            y_base = y_pos  # update base for next bar

    # add pupil data as twin plot
    ax5_pupil = ax5.twinx()
    ax5_pupil.plot([random_avg_pupilarea, pre_avg_pupilarea, post_avg_pupilarea], alpha=0.25, lw=.5, color='olive')
    # plot avg
    ax5_pupil.plot([np.mean(random_avg_pupilarea), np.mean(pre_avg_pupilarea), np.mean(post_avg_pupilarea)],
                color='olive', label='Pupil area', lw=3)
    ax5_pupil.set_ylabel('Avg. pupil area (z-scored)', color='olive', rotation=270, labelpad=15)
    ax5_pupil.tick_params(axis='y', colors='olive')
    ax5_pupil.spines['top'].set_visible(False)
    ax5_pupil.spines['right'].set_visible(True)
    ax5_pupil.set_yticks([0, 0.5, 1])
    ax5_pupil.set_yticklabels([0, 0.5, 1], fontsize=LABEL_FONTSIZE-2)
    ax5_pupil.set_ylim(pupil_ylim)
    ax5_pupil.hlines(0, xmin=0, xmax=2, color='k', linestyle='--', lw=.5)

    area_spacing = 0.1     # vertical spacing between nose bars
    # Perform all pairwise comparisons for pupil area
    for (i, j) in combinations([0, 1, 2], 2):
        data1, data2 = pupilarea_data[i], pupilarea_data[j]
        _, p = ttest_rel(data1, data2) if len(data1) == len(data2) else (np.nan, np.nan)        
        # Plot significance
        if not np.isnan(p) and p < 0.05:
            y_pos = y_base + area_spacing 
            add_significance_bar(axin, i, j, y_pos, p, height=bar_height, color='olive', text_offset=-0.03)
            y_base = y_pos  # update base for next bar

    height_factor = 0.8
    height_offset = +.06
    for ax_selected in [ax0, ax2a]:
        ax_selected.set_xlabel('Saccade-aligned\ntime (s)')
        pos = ax_selected.get_position()
        ax_selected.set_position([pos.x0+.03, pos.y0+height_offset, pos.width*1.2, pos.height*height_factor])
    for ax_selected in [ax1, ax3a]:
        ax_selected.set_xlabel('Non-saccade\naligned time (s)')
        pos = ax_selected.get_position()
        ax_selected.set_position([pos.x0+.02, pos.y0+height_offset, pos.width*1.2, pos.height*height_factor])
    for ax_selected in [ax2a, ax3a]:
        pos = ax_selected.get_position()
        ax_selected.set_position([pos.x0, pos.y0, pos.width, pos.height])
    # shift ax0, ax1, ax4 slightly to left
    for ax_selected in [ax0, ax1, ax4]:
        pos = ax_selected.get_position()
        ax_selected.set_position([pos.x0-.01, pos.y0, pos.width, pos.height])
    pos = ax4.get_position()
    ax4.set_position([pos.x0+.03, pos.y0+height_offset, pos.width, pos.height*height_factor])
    pos = ax5.get_position()
    ax5.set_position([pos.x0+.055, pos.y0+height_offset, pos.width, pos.height*height_factor])


def add_significance_bar(ax, x1, x2, y, p, height=0.01, color='k', linewidth=1, fontsize=12,
                         text_offset=0.01):
    """
    Draws a significance bar between two bars on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    x1, x2 : float
        The x-positions (bar centers) to connect with the significance line.
    y : float
        The base y-position (in data units) where to start the line.
    p : float
        The p-value; determines the text ('***', '**', '*', or 'n.s.').
    height : float
        Height of the bar in data units (fixed, does not depend on y-range).
    color : str
        Color of the bar and text.
    linewidth : float
        Line thickness.
    fontsize : int
        Font size of the significance label.
    """

    # Determine significance symbol
    if p < 0.001:
        significance = '***'
    elif p < 0.01:
        significance = '**'
    elif p < 0.05:
        significance = '*'
    else:
        significance = 'n.s.'
        fontsize = 8

    # Plot horizontal and vertical lines
    ax.plot([x1, x1, x2, x2],
            [y, y + height, y + height, y],
            lw=linewidth, c=color)

    # Add text at midpoint
    ax.text((x1 + x2) / 2, y+text_offset, significance,
            ha='center', va='bottom', color=color, fontsize=fontsize)


def get_centroid_pc():

    dark_dbs = carbachol_in_dark() 
    light_random_dbs = get_light_random_info()
    light_static_dbs = get_light_static_info()
    dbs = [dark_dbs, light_random_dbs, light_static_dbs]
    category = ['dark', 'light_random', 'light_static']

    data_dir = f'{BASE_FOLDER}data/saccade_data/saccade_times.npy'
    saccade_data = np.load(data_dir, allow_pickle=True).item()

    BODYPARTS = [
        "eye(back)",
        "eye(bottom)",
        "eye(front)",
        "eye(top)",
        "lowerlip",
        "mouth",
        "nose(bottom)",
        "nose(r)",
        "nose(tip)",
        "nose(top)",
        "nosebridge",
        "paw",
        "whisker(I)",  
        "whisker(III)",  
        "whisker(II)",  
    ]

    # Initialize a dictionary to store saccade information
    plot_data = {}
    for db_idx, db in enumerate(dbs):
        mouse_list = np.unique([entry['mname'] for entry in db])
        plot_data[category[db_idx]] = {}
        for mname in mouse_list:
            plot_data[category[db_idx]][mname] = {}
            datexp = [entry['datexp'] for entry in db if entry['mname'] == mname][0]

            # Load data
            _, _, _, _, _, _, _, npz_data = load_data(f'{BASE_FOLDER}data/npz_data/', mname, datexp)

            # segment the data into blocks
            block_frames = npz_data['cam_frames_list']
            
            # Get number of blocks in the recording
            blocks = list(npz_data['tcam'].item())
            num_blocks = len(blocks)
            
            if 'light' in category[db_idx]:
                block_idxs = np.arange(1)
            else:
                block_idxs = np.arange(1, num_blocks)

            for blk_idx in block_idxs: 
                blk = blocks[blk_idx].split('blk')[1]
                if category[db_idx] == 'dark' and blk == '1': # no carcabachol in first block
                    continue
                print('Processing block: ', category[db_idx], [mname], [str(blk)])
                plot_data[category[db_idx]][mname][str(blk)] = {}

                pose_x = segment_data(npz_data["pose_x_coord"].T, block_frames, blk_idx)
                pose_y = segment_data(npz_data["pose_y_coord"].T, block_frames, blk_idx)
                whisker_x = np.nanmean(pose_x[:, -3:], axis=1)
                whisker_y = np.nanmean(pose_y[:, -3:], axis=1)
                nose_x = np.nanmean(pose_x[:, 6:10], axis=1)
                nose_y = np.nanmean(pose_y[:, 6:10], axis=1)
                # Do PCA on centroid x, y
                # Combine whisker and nose coordinates into a single array
                whisker_coords = np.column_stack((whisker_x-whisker_x.mean(), whisker_y-whisker_y.mean()))
                nose_coords = np.column_stack((nose_x-nose_x.mean(), nose_y-nose_y.mean()))
                # Fit PCA to whisker coordinates
                pca = PCA(n_components=2)
                whisker_pca = pca.fit_transform(whisker_coords)
                # Fit PCA to nose coordinates
                pca = PCA(n_components=2)
                nose_pca = pca.fit_transform(nose_coords)

                # Calculate the centroid for whiskers and nose
                centroid_data = {
                    'whisker_x': whisker_x,
                    'whisker_y': whisker_y,
                    'whisker_pc1': whisker_pca[:, 0],
                    'whisker_pc2': whisker_pca[:, 1],
                    'nose_x': nose_x,
                    'nose_y': nose_y,
                    'nose_pc1': nose_pca[:, 0],
                    'nose_pc2': nose_pca[:, 1],
                }
                # Append the centroid data to the plot_data dictionary
                plot_data[category[db_idx]][mname][str(blk)] = centroid_data
    return plot_data

def get_saccade_keypoints_relation():

    dark_dbs = carbachol_in_dark() 
    light_random_dbs = get_light_random_info()
    light_static_dbs = get_light_static_info()
    base_folders = [
        f'{BASE_FOLDER}dark_recordings',
        f'{BASE_FOLDER}light_recordings_random',
        f'{BASE_FOLDER}light_recordings_static']
    dbs = [dark_dbs, light_random_dbs, light_static_dbs]
    category = ['dark', 'light_random', 'light_static']


    data_dir = f'{BASE_FOLDER}data/saccade_data/saccade_times.npy'
    saccade_data = np.load(data_dir, allow_pickle=True).item()

    BODYPARTS = [
        "eye(back)",
        "eye(bottom)",
        "eye(front)",
        "eye(top)",
        "lowerlip",
        "mouth",
        "nose(bottom)",
        "nose(r)",
        "nose(tip)",
        "nose(top)",
        "nosebridge",
        "paw",
        "whisker(I)",  
        "whisker(III)",  
        "whisker(II)",  
    ]

    # Initialize a dictionary to store saccade information
    plot_data = {}
    plot_data['pupil_x_start'] = []
    plot_data['pupil_y_start'] = []
    plot_data['pupil_x_end'] = []
    plot_data['pupil_y_end'] = []
    plot_data['saccade_direction'] = []
    plot_data['whisker_x_start'] = []
    plot_data['whisker_y_start'] = []
    plot_data['whisker_x_end'] = []
    plot_data['whisker_y_end'] = []
    plot_data['nose_x_start'] = []
    plot_data['nose_y_start'] = []
    plot_data['nose_x_end'] = []
    plot_data['nose_y_end'] = []
    plot_data['mouse'] = []
    plot_data['whisker_pc1_start'] = []
    plot_data['whisker_pc2_start'] = []
    plot_data['whisker_pc1_end'] = []
    plot_data['whisker_pc2_end'] = []
    plot_data['nose_pc1_start'] = []
    plot_data['nose_pc2_start'] = []
    plot_data['nose_pc1_end'] = []
    plot_data['nose_pc2_end'] = []

    for db_idx, db in enumerate(dbs):
        base_folder = base_folders[db_idx]
        mouse_list = np.unique([entry['mname'] for entry in db])
        for mname in mouse_list:
            datexp = [entry['datexp'] for entry in db if entry['mname'] == mname][0]

            # Load data
            _, _, _, _, _, _, _, npz_data = load_data(f'{BASE_FOLDER}data/npz_data/', mname, datexp)

            # segment the data into blocks
            block_frames = npz_data['cam_frames_list']
            
            # Get number of blocks in the recording
            blocks = list(npz_data['tcam'].item())
            num_blocks = len(blocks)
            
            if 'light' in category[db_idx]:
                block_idxs = np.arange(1)
            else:
                block_idxs = np.arange(1, num_blocks)

            for blk_idx in block_idxs: 
                blk = blocks[blk_idx].split('blk')[1]
                if category[db_idx] == 'dark' and blk == '1': # no carcabachol in first block
                    continue
                print('Processing block: ', category[db_idx], [mname],[str(blk)])

                pose_x = segment_data(npz_data["pose_x_coord"].T, block_frames, blk_idx)
                pose_y = segment_data(npz_data["pose_y_coord"].T, block_frames, blk_idx)
                saccade_info = saccade_data[category[db_idx]][mname][str(blk)]
                whisker_x = np.nanmean(pose_x[:, -3:], axis=1)
                whisker_y = np.nanmean(pose_y[:, -3:], axis=1)
                nose_x = np.nanmean(pose_x[:, 6:10], axis=1)
                nose_y = np.nanmean(pose_y[:, 6:10], axis=1)
                # Do mean subtraction for whisker and nose coordinates
                whisker_x -= whisker_x.mean()
                whisker_y -= whisker_y.mean()
                nose_x -= nose_x.mean()
                nose_y -= nose_y.mean()
                # Do PCA on centroid x, y
                # Combine whisker and nose coordinates into a single array
                whisker_coords = np.column_stack((whisker_x, whisker_y))
                nose_coords = np.column_stack((nose_x, nose_y))
                # Fit PCA to whisker coordinates
                pca = PCA(n_components=2)
                whisker_pca = pca.fit_transform(whisker_coords)
                # Fit PCA to nose coordinates
                pca = PCA(n_components=2)
                nose_pca = pca.fit_transform(nose_coords)

                for sac in saccade_info:
                    start_idx, end_idx, _, _, start_x, start_y, end_x, end_y, _, saccade_direction, _ = sac
                    plot_data['pupil_x_start'].append(start_x)
                    plot_data['pupil_y_start'].append(start_y)
                    plot_data['pupil_x_end'].append(end_x)
                    plot_data['pupil_y_end'].append(end_y)
                    plot_data['saccade_direction'].append(saccade_direction)
                    # Index for saccade and bodypart. use mean of bodyparts to get centroid
                    plot_data['whisker_x_start'].append(whisker_x[start_idx])
                    plot_data['whisker_y_start'].append(whisker_y[start_idx])
                    plot_data['whisker_x_end'].append(whisker_x[end_idx])
                    plot_data['whisker_y_end'].append(whisker_y[end_idx])
                    plot_data['nose_x_start'].append(nose_x[start_idx])
                    plot_data['nose_y_start'].append(nose_y[start_idx])
                    plot_data['nose_x_end'].append(nose_x[end_idx])
                    plot_data['nose_y_end'].append(nose_y[end_idx])
                    plot_data['mouse'].append(mname+'_'+datexp)
                    # Add pc data
                    plot_data['whisker_pc1_start'].append(whisker_pca[start_idx, 0])
                    plot_data['whisker_pc2_start'].append(whisker_pca[start_idx, 1])
                    plot_data['whisker_pc1_end'].append(whisker_pca[end_idx, 0])
                    plot_data['whisker_pc2_end'].append(whisker_pca[end_idx, 1])

                    plot_data['nose_pc1_start'].append(nose_pca[start_idx, 0])
                    plot_data['nose_pc2_start'].append(nose_pca[start_idx, 1])
                    plot_data['nose_pc1_end'].append(nose_pca[end_idx, 0])
                    plot_data['nose_pc2_end'].append(nose_pca[end_idx, 1])

    df = pd.DataFrame(plot_data, columns=plot_data.keys())
    # Calculate deltas (change in positions)
    df['whisker_dx'] = df['whisker_x_end'] - df['whisker_x_start'] #df['whisker_pc1_end'] - df['whisker_pc1_start']
    df['whisker_dy'] = df['whisker_y_end'] - df['whisker_y_start'] #df['whisker_pc2_end'] - df['whisker_pc2_start']
    df['nose_dx'] = df['nose_x_end'] - df['nose_x_start'] #df['nose_pc1_end'] - df['nose_pc1_start']
    df['nose_dy'] = df['nose_y_end'] - df['nose_y_start'] #df['nose_pc2_end'] - df['nose_pc2_start']

    # save df 
    df.to_csv(f'{BASE_FOLDER}data/saccade_data/saccade_keypoints_relation.csv', index=False, header=True, mode='w')

    return df

def plot_saccade_direction_relations(gs):

    if os.path.exists(f'{BASE_FOLDER}data/saccade_data/saccade_keypoints_relation.csv'):
        df = pd.read_csv(f'{BASE_FOLDER}data/saccade_data/saccade_keypoints_relation.csv')
    else:
        df = get_saccade_keypoints_relation()

    # Create subplots
    nested_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs, height_ratios=[1, 1], width_ratios=[1,1], hspace=2.5)
    mouse_list = np.unique(df['mouse'])
    # remove TX150 from mouse list
    mouse_list = mouse_list[mouse_list != 'TX150_2025_02_20']
    bodyparts = ['whisker', 'nose']
    label_names = ['whiskers', 'nose']

    markersize=3
    alpha=.6
    # Separate data by saccade direction
    df_left = df[(df['saccade_direction'] == 'Temporal') & (df['mouse'] == 'TX140_2024_11_06')] #LZ14_2024_06_20
    df_right = df[(df['saccade_direction'] == 'Nasal') & (df['mouse'] == 'TX140_2024_11_06')]

    ### Plotting ###
    ax0 = plt.subplot(nested_gs[0, 0])

    ax0.scatter(df_left['whisker_dx'], df_left['whisker_dy'], color='blue', 
                    alpha=alpha, label='Temporal', s=markersize, edgecolors='none')
    ax0.scatter(df_right['whisker_dx'], df_right['whisker_dy'], color='orange',
                    alpha=alpha, label='Nasal', s=markersize, edgecolors='none')
    ax0.set_xlabel("Whiskers Δx (px)")
    ax0.set_ylabel("Whiskers Δy (px)")
    ax0.set_ylim(-5, 5)
    ax0.set_xlim(-5, 5)
    ax0.set_xticks([])
    # add labels
    ax0.text(-4.75, 4, 'Temporal\nsaccade', color='blue', fontsize=LABEL_FONTSIZE-2)
    ax0.text(1, 4, 'Nasal\nsaccade', color='orange', fontsize=LABEL_FONTSIZE-2)
    ax0.text(8, 8, 'Movement relative to saccade direction',
             ha='center', fontsize=LABEL_FONTSIZE, color='k')
    ax0.set_yticks([-5, 0, 5])
    ax0.set_yticklabels([-5, 0, 5], fontsize=LABEL_FONTSIZE-2)

    ax1 = plt.subplot(nested_gs[1, 0])
    # Make another scatterplot for Y-axis changes
    ax1.scatter(df_left['nose_dx'], df_left['nose_dy'], color='blue', 
                    alpha=alpha, s=markersize, label='Temporal', edgecolors='none')
    ax1.scatter(df_right['nose_dx'], df_right['nose_dy'], color='orange',
                    alpha=alpha, s=markersize, label='Nasal', edgecolors='none')
    ax1.set_xlabel("Nose Δx (px)")
    ax1.set_ylabel("Nose Δy (px)")
    ax1.set_ylim(-5, 5)
    ax1.set_xlim(-5, 5)
    ax1.set_yticks([-5, 0, 5])
    ax1.set_yticklabels([-5, 0, 5], fontsize=LABEL_FONTSIZE-2)

    # Adjust positions
    y_adjust = -.19
    x_adjust = -.13
    pos = ax0.get_position()
    ax0.set_position([pos.x0+x_adjust, pos.y0+y_adjust, pos.width*2, pos.height*2])
    pos = ax1.get_position()
    ax1.set_position([pos.x0+x_adjust, pos.y0+y_adjust+.05, pos.width*2, pos.height*2])
    ax0.set_aspect('equal', adjustable='box')
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot avg per mouse
    for bp_idx, bodypart in enumerate(bodyparts):
        ax = plt.subplot(nested_gs[bp_idx, 1])

        left_means = []
        right_means = []
        for mouse in mouse_list:
            df_mouse = df[df['mouse'] == mouse]
            df_left_mouse = df_mouse[df_mouse['saccade_direction'] == 'Temporal']
            df_right_mouse = df_mouse[df_mouse['saccade_direction'] == 'Nasal']
            left_mean = [np.nanmean(df_left_mouse['{}_dx'.format(bodypart)]), np.nanmean(df_left_mouse['{}_dy'.format(bodypart)])]
            right_mean = [np.nanmean(df_right_mouse['{}_dx'.format(bodypart)]), np.nanmean(df_right_mouse['{}_dy'.format(bodypart)])]
            left_means.append(left_mean)
            right_means.append(right_mean)
        left_means = np.array(left_means)
        right_means = np.array(right_means) 
        # Perform paired t-test for dx and dy
        t_dx, p_dx = ttest_rel(left_means[:, 0], right_means[:, 0], nan_policy='omit')
        t_dy, p_dy = ttest_rel(left_means[:, 1], right_means[:, 1], nan_policy='omit')

        for i in range(len(mouse_list)):
            ax.plot([left_means[i, 0], right_means[i, 0]], [left_means[i, 1], right_means[i, 1]], 'k-', alpha=0.2)
            ax.scatter(left_means[i, 0], left_means[i, 1], color='blue', label='Temporal' if i == 0 else "", s=3)
            ax.scatter(right_means[i, 0], right_means[i, 1], color='orange', label='Nasal' if i == 0 else "", s=3)
        ax.set_xlabel('Mean {} Δx'.format(label_names[bp_idx]))
        ax.set_ylabel('Mean {} Δy'.format(label_names[bp_idx]))
        #ax.set_ylabel('Mean {} Δy'.format(label_names[bp_idx]))
        ax.axhline(0, color='k', linestyle='--', lw=0.5)
        ax.axvline(0, color='k', linestyle='--', lw=0.5)
        ax.set_xlim(-5, 5)
        #ax.set_ylim(-2, 2)
        # set position to lower and increase width
        x_adjust = -.065
        pos = ax.get_position()
        if bp_idx == 0:
            ax.set_position([pos.x0+x_adjust, pos.y0+y_adjust, pos.width*.55, pos.height*2])
        else:
            ax.set_position([pos.x0+x_adjust, pos.y0+y_adjust+.05, pos.width*.55, pos.height*2])
        #ax.set_aspect('equal', adjustable='box')
        if bp_idx == 0:
            ax.set_xticks([])

        # Add a histogram inset to show distribution of dx for left and right oriiente
        ax_inset = ax.inset_axes([0., 1.05, 1, 0.15])
        ax_inset.hist(left_means[:,0], bins=10, alpha=0.3, color='blue', label='Temporal')
        ax_inset.hist(right_means[:,0], bins=10, alpha=0.3, color='orange', label='Nasal')
        ax_inset.set_yticks([])
        ax_inset.set_xticks([])
        ax_inset.spines['left'].set_visible(False)    
        ax_inset.set_xlim(-5, 5)
        # Add significance bar 
        max_height = ax_inset.get_ylim()[1]
        height_factor = max_height * 0.1
        ax_inset.plot([0, 0, 1, 1], [max_height, max_height+height_factor, max_height+height_factor, max_height],   
                    color='k', lw=1)    
        p_text = '***' if p_dx < 0.001 else '**' if p_dx < 0.01 else '*' if p_dx < 0.05 else f'p={p_dx:.2f}'
        ax_inset.text(0.5, max_height*0.9+height_factor, p_text,
                    ha='center', va='bottom', fontsize=LABEL_FONTSIZE-2, color='k')

        # Add another histogram inset to show distribution of dy for left and right oriented
        ax_inset2 = ax.inset_axes([1.05, 0., 0.15, 1])
        ax_inset2.hist(left_means[:,1], bins=10, alpha=0.3, color='blue', orientation='horizontal')
        ax_inset2.hist(right_means[:,1], bins=10, alpha=0.3, color='orange', orientation='horizontal')
        ax_inset2.set_yticks([])
        ax_inset2.set_xticks([])
        #ax_inset2.set_xlim(-3, 3)
        #ax_inset2.set_ylim(-5, 5)
        ax_inset2.spines['bottom'].set_visible(False)
        # Add significance bar
        max_width = ax_inset2.get_xlim()[1]
        width_factor = max_width * 0.1
        ax_inset2.plot([max_width, max_width+width_factor, max_width+width_factor, max_width],
                    [0, 0, 1, 1], color='k', lw=1)
        p_text = '***' if p_dy < 0.001 else '**' if p_dy < 0.01 else '*' if p_dy < 0.05 else f'n.s.'
        ax_inset2.text(max_width+width_factor, 0.5, p_text,
                    ha='left', va='center', fontsize=LABEL_FONTSIZE-2, color='k', rotation=-90)

def check_mouse_saccade_direction_relations():
    if os.path.exists(f'{BASE_FOLDER}data/saccade_data/saccade_keypoints_relation.csv'):
        df = pd.read_csv(f'{BASE_FOLDER}data/saccade_data/saccade_keypoints_relation.csv')
    else:
        df = get_saccade_keypoints_relation()

    # Create subplots
    mouse_list = np.unique(df['mouse'])
    bodyparts = ['whisker', 'nose']
    label_names = ['whiskers', 'nose']

    markersize=5
    alpha=1
    fig, ax = plt.subplots(len(mouse_list), 2, figsize=(6, 2.5*len(mouse_list)))
    from scipy.stats import ttest_ind

    for m_idx, mouse in enumerate(mouse_list):
        # Separate data by saccade direction
        df_left = df[(df['saccade_direction'] == 'Temporal') & (df['mouse'] == mouse)] 
        df_right = df[(df['saccade_direction'] == 'Nasal') & (df['mouse'] == mouse)]

        ### Plotting ###
        ax0 = ax[m_idx, 0]

        ax0.scatter(df_left['whisker_dx'], df_left['whisker_dy'], color='blue', 
                        alpha=alpha, label='Temporal', s=markersize, edgecolors='none')
        ax0.scatter(df_right['whisker_dx'], df_right['whisker_dy'], color='orange',
                        alpha=alpha, label='Nasal', s=markersize, edgecolors='none')
        ax0.set_xlabel("Whiskers Δx (px)")
        ax0.set_ylabel("Whiskers Δy (px)")
        ax0.set_ylim(-5, 5)
        ax0.set_xlim(-5, 5)
        ax0.set_xticks([])
        # add labels
        if m_idx == 0:
            ax0.text(-4.75, 4, 'Temporal\nsaccade', color='blue', fontsize=LABEL_FONTSIZE-2)
            ax0.text(1, 4, 'Nasal\nsaccade', color='orange', fontsize=LABEL_FONTSIZE-2)
        ax0.set_yticks([-5, 0, 5])
        ax0.set_yticklabels([-5, 0, 5], fontsize=LABEL_FONTSIZE-2)
        ax0.text(7, 6, '{m}'.format(m=mouse),
            ha='center', fontsize=LABEL_FONTSIZE, color='k')
        # perform ttest and add significance
        t_dx, p_dx = ttest_ind(df_left['whisker_dx'], df_right['whisker_dx'])
        t_dy, p_dy = ttest_ind(df_left['whisker_dy'], df_right['whisker_dy'])
        y_base = 4.5
        bar_height = 0.2
        add_significance_bar(ax0, -4, 4, y_base, p_dx, height=bar_height, color='k', text_offset=0.1)
        #make p_dy bar vertical
        bar_height = 9
        add_significance_bar(ax0, 4.5, 4.5, -4.5, p_dy, height=bar_height, color='k', text_offset=5)

        ax1 = ax[m_idx, 1]
        # Make another scatterplot for Y-axis changes
        ax1.scatter(df_left['nose_dx'], df_left['nose_dy'], color='blue', 
                        alpha=alpha, s=markersize, label='Temporal', edgecolors='none')
        ax1.scatter(df_right['nose_dx'], df_right['nose_dy'], color='orange',
                        alpha=alpha, s=markersize, label='Nasal', edgecolors='none')
        ax1.set_xlabel("Nose Δx (px)")
        ax1.set_ylabel("Nose Δy (px)")
        #ax1.set_ylim(-5, 5)
        #ax1.set_xlim(-5, 5)
        ax1.set_yticks([-5, 0, 5])
        ax1.set_yticklabels([-5, 0, 5], fontsize=LABEL_FONTSIZE-2)
        # perform ttest and add significance
        t_dx, p_dx = ttest_ind(df_left['whisker_dx'], df_right['whisker_dx'])
        t_dy, p_dy = ttest_ind(df_left['whisker_dy'], df_right['whisker_dy'])
        y_base = 4.5
        bar_height = 0.2
        add_significance_bar(ax1, -4, 4, y_base, p_dx, height=bar_height, color='k', text_offset=0.1)
        #make p_dy bar vertical
        bar_height = 9
        add_significance_bar(ax1, 4.5, 4.5, -4.5, p_dy, height=bar_height, color='k', text_offset=5)
        
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()


def plot_prediction_traces(gs):
    # Load trace data to plot in fig.3 
    data = np.load(f"{BASE_FOLDER}data/saccade_data/pred_traces_zscore.npy", allow_pickle=True).item()
    whisker_color = "#c92814ff"
    nose_color = "#6bfbbaff"
    pupil_x_color = 'darkgreen'
    pupil_y_color = 'lightgreen'

    nested_gs = GridSpecFromSubplotSpec(6, 1, subplot_spec=gs, hspace=.1)

    for example in range(3):
        if example == 0:
            input_name = 'whisker'
            output_name = 'nose'
        else:
            input_name = 'whisker'
            output_name = 'pupil'
        input_idx = [idx for idx in range(len(data['input_group'])) if data['input_group'][idx] == input_name]
        output_idx = [idx for idx in range(len(data['output_group'])) if data['output_group'][idx] == output_name]

        # Plot the selected input and output trace
        if input_name == 'whisker':
            input_color = [whisker_color, whisker_color]
        elif input_name == 'nose':
            input_color = [nose_color, nose_color]
        elif input_name == 'pupil':
            input_color = [pupil_x_color, pupil_y_color]
        if output_name == 'pupil':
            output_color = [pupil_x_color, pupil_y_color]
        elif output_name == 'nose':
            output_color = [nose_color, nose_color]
        elif output_name == 'whisker':
            output_color = [whisker_color, whisker_color]
        lw = 1

        base_row = example * 2
        ax0 = plt.subplot(nested_gs[base_row])
        rand_idx = np.random.randint(len(data['inputs_original']))
        chunk_start = np.random.randint(0, data['inputs_original'][rand_idx].shape[0] - 1000)
        chunk_idx = np.arange(chunk_start, chunk_start+1000)

        in_data =  data['inputs_original'][rand_idx] #data['inputs_original'][select_idx].reshape(data['inputs_original'][select_idx].shape[0], -1, 2)
        ax0.plot(in_data[chunk_idx, 0], color=input_color[0], label=input_name, lw=lw, linestyle='-')
        ax0.plot(in_data[chunk_idx, 1], color=input_color[1], label=input_name, lw=lw, linestyle=':')
        ax0.text(0, 1, "Input group: {} x,y".format(input_name), transform=ax0.transAxes, fontsize=LABEL_FONTSIZE-1)
        ax0.set_xticks([])
        ax0.set_yticks([])
        remove_spines(ax0)

        ax1 = plt.subplot(nested_gs[base_row+1])
        out_data = data['test_data'][rand_idx]#.reshape(data['test_data'][select_idx].shape[0], -1, 2)
        ax1.plot(out_data[chunk_idx, 0], color=output_color[0], label=output_name, lw=lw)
        pred_data = data['pred_data'][rand_idx]#.reshape(data['pred_data'][select_idx].shape[0], -1, 2)
        ax1.plot(pred_data[chunk_idx, 0], color='grey', label=output_name, linestyle='-', lw=lw-.4)
        ax1.text(0, 1, "Predicted group: {} x".format(output_name), transform=ax1.transAxes, fontsize=LABEL_FONTSIZE-1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        remove_spines(ax1)
        # create invisible proxy lines (no visible handle)
        proxy_target = Line2D([], [], linestyle='-', color='k')
        proxy_pred = Line2D([], [], linestyle='-', color='grey')
        leg = ax1.legend(
            [proxy_target, proxy_pred],
            ['Target', 'Prediction'],
            frameon=False,
            loc='upper right',
            fontsize=10,
            ncol=2,
            bbox_to_anchor=(1, 1.4),
            handletextpad=0,    # tiny padding if you want some gap
            handlelength=0,
        )
        # change color of text
        for text in leg.get_texts():
            if text.get_text() == 'Prediction':
                text.set_color('grey')
            else:
                text.set_color(output_color[0])
                        
        # Move position of plot to right
        y_adjust = -.11 * ((example+1)*.65)
        x_adjust = -.12
        pos = ax0.get_position()
        ax0.set_position([pos.x0+x_adjust, pos.y0+y_adjust, pos.width*1.4, pos.height])
        pos = ax1.get_position()
        ax1.set_position([pos.x0+x_adjust, pos.y0+y_adjust-.015, pos.width*1.4, pos.height])
        
        # --- Add arrow between input and output subplots ---
        pos_in = ax0.get_position()
        pos_out = ax1.get_position()

        # Compute midpoint x (same horizontal center as the subplots)
        arrow_x = pos_in.x0 + pos_in.width / 2
        # Arrow starts just below input and ends just above output
        arrow_y_start = pos_in.y0 + 0.01 #pos_out.y0 + pos_out.height 
        arrow_y_end = pos_in.y0 - 0.015 

        # Draw arrow in figure coordinates via the parent Axes' figure
        ax0.figure.add_artist(plt.Arrow(
            x=arrow_x, y=arrow_y_start,
            dx=0, dy=arrow_y_end - arrow_y_start,
            width=0.015, color='k', alpha=0.7
        ))

def segment_data(data, frames_per_folder, idx):
    idx = np.arange(sum(frames_per_folder[:idx]), sum(frames_per_folder[:idx+1])) 
    return data[idx]

def load_data(base_folder, mname, datexp):
    npz_data = np.load(f"{base_folder}/{mname}_{datexp}_{1}.npz", allow_pickle=True)
    spks = npz_data['spks']
    xpos = npz_data['xpos']
    ypos = npz_data['ypos']
    tcams = npz_data['tcam'].item()
    tneurals = npz_data['tneural'].item()
    cumframes_neural = np.hstack(([0], np.cumsum(npz_data['neural_frames_list'])))
    cumframes_cam = np.hstack(([0], np.cumsum(npz_data['cam_frames_list'])))
    
    return spks, xpos, ypos, tcams, tneurals, cumframes_neural, cumframes_cam, npz_data       

def carbachol_in_dark():
    """ if mname, sess, blk and rewType are all empty, it will output all session
        if only want first session from all all animals, sess=[1]
    """
    # all the following recordings have direction: nasal (left side of video) to temporal (right side of video)
    print('Sessions from carbachol_in_dark >>>')
    db=[]
    db.append({'sess#':1,'mname':'TX120','datexp':'2024_06_19','blk':'1','is2p':1,'rewType':'None','exptype':'baseline','Note':'3Hz'})
    db.append({'sess#':2,'mname':'TX120','datexp':'2024_06_19','blk':'3','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.4', 'speed_threshold':'99.5'})
    db.append({'sess#':3,'mname':'TX120','datexp':'2024_06_19','blk':'4','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.5', 'speed_threshold':'99.7'})
    
    db.append({'sess#':1,'mname':'TX121','datexp':'2024_06_19','blk':'1','is2p':1,'rewType':'None','exptype':'baseline','Note':'3Hz'})
    db.append({'sess#':2,'mname':'TX121','datexp':'2024_06_19','blk':'3','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.7'})
    db.append({'sess#':3,'mname':'TX121','datexp':'2024_06_19','blk':'4','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.7'})  
    
    db.append({'sess#':1,'mname':'TX123','datexp':'2024_06_20','blk':'1','is2p':1,'rewType':'None','exptype':'baseline','Note':'3Hz'})
    db.append({'sess#':2,'mname':'TX123','datexp':'2024_06_20','blk':'2','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'1.5', 'speed_threshold':'99.5'})  
      
    db.append({'sess#':1,'mname':'LZ14','datexp':'2024_06_20','blk':'1','is2p':1,'rewType':'None','exptype':'baseline','Note':'3Hz'})
    db.append({'sess#':2,'mname':'LZ14','datexp':'2024_06_20','blk':'2','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})  
    
    db = np.array(db)
    return db

def get_light_static_info():
    # all the following recordings have direction: temporal (left side of video) to nasal (right side of video)
    print("Sessions from get_light_static_info >>>")
    db=[]
    db.append({'sess#':1,'mname':'TX143','datexp':'2024_10_28','blk':'1', 'blink_baselined_threshold':'1.5', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'TX143','datexp':'2024_10_28','blk':'3'})
    
    db.append({'sess#':1,'mname':'TX139','datexp':'2024_10_31','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'TX139','datexp':'2024_10_31','blk':'2'})

    db.append({'sess#':1,'mname':'LZ16','datexp':'2024_11_04','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.9'})
    db.append({'sess#':2,'mname':'LZ16','datexp':'2024_11_04','blk':'2'})
    
    db.append({'sess#':1,'mname':'TX140','datexp':'2024_11_06','blk':'1', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'TX140','datexp':'2024_11_06','blk':'2'})

    db.append({'sess#':1,'mname':'FXa1','datexp':'2026_01_15','blk':'1', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'FXa1','datexp':'2026_01_15','blk':'2'})

    db.append({'sess#':1,'mname':'FXa8','datexp':'2026_01_16','blk':'1', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'FXa8','datexp':'2026_01_16','blk':'2'})

    db.append({'sess#':1,'mname':'FXa10','datexp':'2026_01_16','blk':'1', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'FXa10','datexp':'2026_01_16','blk':'2'})

    db = np.array(db)
    return db

def get_light_random_info():
    print("Sessions from get_light_random_info >>>")
    db=[]
    # random stim data
    # all the following recordings have direction: temporal (left side of video) to nasal (right side of video)
    db.append({'mname':'LZ16','datexp':'2024_07_29','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.9'})
    db.append({'mname':'TX139','datexp':'2024_08_10','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})
    db.append({'mname':'TX143','datexp':'2024_08_10','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.7'})
    db.append({'mname':'TX144','datexp':'2024_08_10','blk':'1', 'blink_baselined_threshold':'.4', 'speed_threshold':'99.5'})
    db.append({'mname':'FXa8','datexp':'2026_01_13','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})
    db.append({'mname':'FXa9','datexp':'2026_01_15','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})
    db.append({'mname':'FXa10','datexp':'2026_01_13','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})    
    db = np.array(db)
    return db