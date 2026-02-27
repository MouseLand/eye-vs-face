from fig3 import get_light_static_info, load_data
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
import sys
sys.path.insert(0, '/home/asyeda/Facemap/facemap')
from facemap import pupil
from itertools import combinations
import pandas as pd
import seaborn as sns
import cv2

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


# Set folder
BASE_FOLDER = 'set_your_base_folder_here/' 
mname = 'TX143'
pupil_xlim = [66500, 69700]  # x-axis limits for the plot
dbs = get_light_static_info()
datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]
npz_data = None

def set_path(data_path):
    global BASE_FOLDER, npz_data
    BASE_FOLDER = data_path
    mname = 'TX143'
    _, _, _, _, _, _, _, npz_data = load_data(f"{BASE_FOLDER}data/npz_data/", mname, datexp)


def plot(data_path):
    
    set_path(data_path)

    fig = plt.figure(figsize=(14, 7), dpi=300)
    gs = GridSpec(2, 3, figure=fig, hspace=.25, wspace=0.35, height_ratios=[.8,1.2])

    plot_saccade_traces(gs[0, 0])
    plot_eye_axis(gs[0, 1])
    plot_saccade_traces(gs[0, 2], polar=True)
    plot_saccade_duration_hist(gs[1,0])
    plot_amp_vs_vel(gs[1,1])
    plot_saccade_directions(gs[1,2])

    # save figure
    fig.tight_layout()
    plt.savefig(f'{BASE_FOLDER}figures/saccade.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_saccade_traces(gs, polar=False):
    # Load the data
    if polar:
        pupil_x = npz_data['pupil_x_polar']
        pupil_y = npz_data['pupil_y_polar']
    else:
        pupil_x = npz_data['pupil_x_small']
        pupil_y = npz_data['pupil_y_small']

    # Plot pupil x and y
    pupil_x_color = 'darkgreen'
    pupil_y_color = 'lightgreen'
    if polar:
        label_x, label_y = 'Azimuth', 'Elevation'
    else:
        label_x, label_y = 'X', 'Y'

    ax1 = plt.subplot(gs)
    ax1.plot(pupil_x, color=pupil_x_color, label=label_x, lw=2)
    ax1.plot(pupil_y, color=pupil_y_color, label=label_y, lw=2)
    ax1.set_xlim(pupil_xlim)
    ax1.set_yticks([])
    ax1.set_xticks([])
    if polar:
        ax1.set_ylim(70, 100)
    else:
        ax1.set_ylim(15, 32)
    ax1.set_anchor('C')              # center axes
    ax1.set_position(ax1.get_position())

    proxy_azimuth = Line2D([], [], linestyle='--', color=pupil_x_color)
    proxy_elevation = Line2D([], [], linestyle='--', color=pupil_y_color)
    leg = ax1.legend(
                [proxy_azimuth, proxy_elevation],
                [label_x, label_y],
                frameon=False,
                fontsize=16,
                ncol=2,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.01),            # fixed axes-relative position
                bbox_transform=ax1.transAxes,           # CRITICAL
                handlelength=0,             # removes handle space
                handletextpad=0.2           # tiny padding if you want some gap
            )
    # Add y scale bar to denote pixels or degrees
    if polar:
        scale_bar_length = 10
        scale_bar_label = '10°'
    else:
        scale_bar_length = 5
        scale_bar_label = '5 px'
    # Get current limits
    ymin, ymax = ax1.get_ylim()
    xmin, xmax = ax1.get_xlim()

    # Plot scale bar 
    x_bar = xmax + 0.02 * (xmax - xmin)     # small inset from right
    y_bar_start = ymin + 0.1 * (ymax - ymin)

    # Draw scale bar (true data length)
    ax1.plot(
        [x_bar, x_bar],
        [y_bar_start, y_bar_start + scale_bar_length],
        color='k',
        lw=2,
        clip_on=False
    )

    # Label
    ax1.text(
        x_bar + 0.01 * (xmax - xmin),
        y_bar_start + scale_bar_length / 2,
        scale_bar_label,
        va='center',
        fontsize=12
    )

    for text in leg.get_texts():
        if text.get_text() == 'X' or text.get_text() == 'Azimuth':
            text.set_color(pupil_x_color)
        else:
            text.set_color(pupil_y_color)

    # Add saccade times 
    saccade_times = np.load(f'{BASE_FOLDER}data/saccade_data/saccade_times.npy', allow_pickle=True).item()
    for saccade_time in saccade_times['light_static'][mname]['1']:
        #ax1.axvspan(saccade_time[0], saccade_time[1], color='green', alpha=0.75)
        ax1.axvline(saccade_time[0], color='green', linestyle='dashed', lw=0.7)

    # remove spines
    for spine in ax1.spines.values():
        spine.set_visible(False)


def plot_eye_axis(gs):
    ax = plt.subplot(gs)
    # Load pupil data
    datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]
    axis_info = np.load(f"{BASE_FOLDER}data/mouse_data/{mname}/{datexp}/{mname}_{datexp}_eyekp.npy", allow_pickle=True).item()
    frame_idx = np.random.randint(pupil_xlim[0], pupil_xlim[1])
    print("frame idx:", frame_idx)
    
    # Load video 
    video_path = f"{BASE_FOLDER}data/mouse_data/{mname}/{datexp}/{mname}_{datexp}_{1}_small.mp4"
    frame = get_frame_by_index(video_path, frame_idx=frame_idx)
    
    # Load pupil data
    pupil_x = npz_data['pupil_x_small']
    pupil_y = npz_data['pupil_y_small']

    proc_filepath = f"{BASE_FOLDER}data/mouse_data/{mname}/{datexp}/{mname}_{datexp}_{1}_small_proc.npy"
    proc_data = np.load(proc_filepath, allow_pickle=True).item()
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
    u, sv = proc_data['pupil'][0]['axdir'][frame_idx], proc_data['pupil'][0]['axlen'][frame_idx]
    n = 100  # Number of points around ellipse
    p = np.linspace(0, 2 * np.pi, n)[:, np.newaxis]
    xy = np.concatenate((np.cos(p), np.sin(p)), axis=1) * (sv**0.5) @ u
    xy += mu

    # Plot 
    img = cv2.convertScaleAbs(frame, alpha=6, beta=2.5)
    ax.imshow(img, cmap='grey', vmin=0, vmax=90)
    ax.scatter(axis_info['c1'][0], axis_info['c1'][1], color='red')
    ax.scatter(axis_info['c2'][0], axis_info['c2'][1], color='red')
    # dotted line connecting them
    ax.plot(
        [axis_info['c1'][0], axis_info['c2'][0]],
        [axis_info['c1'][1], axis_info['c2'][1]],
        linestyle=':',
        color='w', lw=1.5
    )
    ax.scatter(xy[:, 1]+crop_x[0], xy[:, 0]+crop_y[0], c='olive', s=1.2, 
               marker='+', zorder=5, edgecolors='none', lw=2)
    ax.scatter(mu[1]+crop_x[0], mu[0]+crop_y[0], c='green', s=50, marker='+', lw=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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

def plot_saccade_workflow(gs):
    # image file
    image_path = f'{BASE_FOLDER}images/saccade_workflow.png'
    ax = plt.subplot(gs)
    img = plt.imread(image_path)
    ax.imshow(img)
    ax.axis('off')  # Hide the axes
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*.9, pos.height])

def plot_saccade_directions(gs):
    # Load data
    data_dir = f'{BASE_FOLDER}data/saccade_data/saccade_data.npz'
    saccade_data = np.load(data_dir, allow_pickle=True)
    beta_x = saccade_data['beta_x']
    beta_y = saccade_data['beta_y']
    direction = saccade_data['saccade_directions']

    # Plot saccade directions
    ax = plt.subplot(gs, projection='polar')

    # Calculate the angles from beta_x and beta_y
    beta_x = np.array(beta_x)
    beta_y = np.array(beta_y)
    # Calculate angles in radians
    angles_rad = np.arctan2(beta_y, beta_x)  # This gives angles in [-π, π]

    # Make histogram of angles
    bins = np.linspace(0, 2 * np.pi, 20)
    counts, bin_edges = np.histogram(angles_rad, bins=bins)
    probabilities = counts / np.sum(counts)  

    bars = ax.bar(bin_edges[:-1], probabilities, width=np.diff(bin_edges), color='grey', edgecolor='k', alpha=0.7, align='center')

    # Add probability values on top of each bar
    for angle, prob, bar in zip(bin_edges[:-1], probabilities, bars):
        if prob > 0.2:
            height = bar.get_height()
            ax.text(
                angle,  # Angle position
                height + 0.05,  # Slightly above the bar
                f"{prob:.2f}",  # Format probability to 2 decimal places
                ha='center',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                fontsize=10,  # Font size
            )

    ax.set_xticks(np.pi / 2 * np.array([0, 1, 2, 3]))  # 0° (right), 90° (up), 180° (left), 270° (down)
    ax.set_xticklabels(['Temporal', 'Dorsal', 'Nasal', 'Ventral'], fontsize=12, y=-0.16, x=0)  # Adjust y position for better visibility
    ax.set_yticklabels([])  # Remove radial axis numbers
    #ax.set_title("Saccade Direction Probabilities")

def plot_saccade_duration_hist(gs):
    # Load data
    data_dir = f'{BASE_FOLDER}data/saccade_data/saccade_data.npz'
    print("Loading from:", data_dir)
    saccade_info = np.load(data_dir, allow_pickle=True)
    freq = saccade_info['frequencies']
    cond = saccade_info['condition']
    print(saccade_info.keys())

    ax = plt.subplot(gs)

    cond_colors = {
        'light_static':  '#008080',
        'light_random': '#6a0dad',
        'dark': "#8080ff"
    }

    # Map conditions to x positions
    cond_map = {
        'light_static': 1,
        'light_random': 2,
        'dark': 3
    }

    # Scatter individual points (colored by condition)
    for c, x in cond_map.items():
        idx = (cond == c)
        x_jitter = x + np.random.normal(0, 0.08, size=np.sum(idx))

        ax.scatter(
            x_jitter,
            freq[idx],
            color=cond_colors[c],
            alpha=1,
            s=30,
            zorder=2
        )

    # Plot horizontal mean lines (no dots, no connections)
    for c, x in cond_map.items():
        mean_val = np.mean(freq[cond == c])

        ax.hlines(
            mean_val,
            x - 0.15,
            x + 0.15,
            color=cond_colors[c],
            lw=3,
            zorder=3
        )

    # Axis formatting
    ax.set_xlim(0.5, 3.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Light(static)', 'Light(random)', 'Dark'], fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Saccades / min', fontsize=LABEL_FONTSIZE)
    ax.set_ylim(np.min(saccade_info['frequencies'])-.5, np.max(saccade_info['frequencies']) + 1)
    print("max freq:", np.max(saccade_info['frequencies']))
    print("min freq:", c) 
    ax.set_yticks(np.arange(0, np.max(saccade_info['frequencies']) + 2, 5))

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Match your layout tweak
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])


def plot_amp_vs_vel(gs):
    # Load data
    data_dir = f'{BASE_FOLDER}data/saccade_data/saccade_data.npz'
    saccade_data = np.load(data_dir, allow_pickle=True)
    saccade_amplitudes = saccade_data['saccade_amplitudes']
    peak_velocities = saccade_data['peak_velocities']

    # Plot amplitude vs velocity
    ax = plt.subplot(gs)
    ax.scatter(peak_velocities, saccade_amplitudes, color='grey', alpha=0.7, s=2)
    ax.set_ylabel('Saccade Amplitude (px)')
    ax.set_xlabel('Saccade speed (px/s)')
    pos = ax.get_position()
    ax.set_position([pos.x0+.01, pos.y0, pos.width, pos.height])
    #ax.axis('square')


def plot_saccade_durations(gs):
    # Load data
    data_dir = f'{BASE_FOLDER}data/saccade_data/saccade_data.npz'
    saccade_data = np.load(data_dir, allow_pickle=True)
    saccade_durations = saccade_data['saccade_durations']

    # Plot saccade durations
    ax = plt.subplot(gs[0, 1])
    ax.hist(saccade_durations, bins=30, color='grey', edgecolor='k', alpha=0.7)
    ax.set_xlabel('Saccade Duration (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title("Saccade Duration Distribution")
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*1.5, pos.height])
