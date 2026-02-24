from fig_utils import *
import matplotlib.pyplot as plt 
import os, sys
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import zscore
from scipy.io import loadmat
sys.path.insert(0, '..')
import helper_functions
from rastermap import Rastermap, utils
from scipy.stats import ttest_rel
import cv2
from glob import glob
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
sys.path.insert(0, '/home/asyeda/Facemap/facemap')
from facemap import pupil
import pupil_dark_helpers as pdh
import nonlinear_modelv3
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
from tqdm import tqdm

LABEL_FONTSIZE = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = LABEL_FONTSIZE
plt.rcParams['axes.titlesize'] = LABEL_FONTSIZE
plt.rcParams['axes.labelsize'] = LABEL_FONTSIZE
plt.rcParams['xtick.labelsize'] = LABEL_FONTSIZE-2
plt.rcParams['ytick.labelsize'] = LABEL_FONTSIZE-2
plt.rcParams['legend.fontsize'] = LABEL_FONTSIZE-2

pupil_area_label_color = 'olive'
pupil_label_color = 'green'
face_eye_label_color = 'darkorange'
face_eye_pupil_label_color = 'red'
rand_stat_colors = ['#6a0dad', '#008080']
rand_stat_color_names = ['random', 'static']
pupil_x_color = 'darkgreen'
pupil_y_color = 'lightgreen'
BASE_FOLDER = "/media/Atika/eye_movement_paper/pupil_analysis/"

def plot():
    fig = plt.figure(figsize=(14, 9), dpi=300)
    gs = GridSpec(3, 8, figure=fig, hspace=.1, wspace=.55, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1, .75, .75])

    video_path = f"{BASE_FOLDER}light_recordings_random/TX139_2024_08_10_1_full.mp4"
    small_video_path = f"{BASE_FOLDER}light_recordings_random/TX139_2024_08_10_1_small.mp4"
    small_proc_filepath = f"{BASE_FOLDER}light_recordings_random/TX139_2024_08_10_1_small_proc.npy"
    full_proc_filepath = f"{BASE_FOLDER}light_recordings_random/TX139_2024_08_10_1_full_proc.npy"
    plot_exp_illustration(plt.subplot(gs[0, 0]), f"{BASE_FOLDER}paper_figures/exp_illustration.png")
    plot_recording_regions(plt.subplot(gs[0, 1]))
    plot_mouse_face(plt.subplot(gs[0, 2]), video_path, frame_idx=25366, crop_x=[155, 370], crop_y=[40, 220], proc_filepath=full_proc_filepath)
    plot_mouse_eye(plt.subplot(gs[0, 3]), small_video_path, frame_idx=25366, proc_filepath=small_proc_filepath)
    plot_network(plt.subplot(gs[0, 4:]), filepath=f'{BASE_FOLDER}paper_figures/network_fig1.png')
    # Second and third rows
    stim_presentation(gs[1,0], f'{BASE_FOLDER}/images/leaves_test.jpg')
    plot_tuning_curve(gs[1, 1])
    plot_train_test_tuning_scatter(gs[2, 0])
    plot_correlation_hist(gs[2, 1])
    #raster_plot(fig, gs[1,2:5], 'static')
    raster_plot_light_predictions(fig, gs[1:,2:4], 'static')
    raster_plot_light_predictions(fig, gs[1:,4:6], 'random')
    #raster_plot(fig, gs[2,2:5], 'random')
    scatter_plot(fig, gs[1,6], 'static')
    scatter_plot(fig, gs[1,7], 'random')
    line_plot(fig, gs[2,6], 'static')
    line_plot(fig, gs[2,7], 'random')

    # save figure
    fig.tight_layout()
    plt.savefig(f'{BASE_FOLDER}paper_figures/figure1.png', dpi=300, bbox_inches='tight')
    plt.show()

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

def plot_mouse_face(ax, video_path, frame_idx, crop_x, crop_y, proc_filepath):

    frame = get_frame_by_index(video_path, frame_idx)
    # Crop the image
    img = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

    # Plot the image
    img = cv2.convertScaleAbs(img, alpha=1, beta=1.5)
    ax.imshow(img, cmap='gray')

    # Add pupil crop box
    proc_data = np.load(proc_filepath, allow_pickle=True).item()
    xrange = proc_data['rois'][1]['xrange']
    yrange = proc_data['rois'][1]['yrange']
    # plot a line from xrange[0], yrange[0] to xrange[-1], yrange[0]
    rect = plt.Rectangle((xrange[0]-crop_x[0], yrange[0]-crop_y[0]), 
                         xrange[-1]-xrange[0], yrange[-1]-yrange[0], 
                         linewidth=1, edgecolor='green', facecolor='none')
    ax.add_patch(rect)

    ax.axis('off')
    ax.set_title("Face view", fontsize=LABEL_FONTSIZE)
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+.02, pos[1]-.02, pos[2]*1.3, pos[3]*1.3])

def plot_mouse_eye(ax, video_path, frame_idx, proc_filepath):
    proc_data = np.load(proc_filepath, allow_pickle=True).item()
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
    """
    area = np.pi * (sig[0] * sig[1]) ** 0.5
    xy = xy[xy[:, 0] >= 0, :]
    xy = xy[xy[:, 0] < yrange.size, :]
    xy = xy[xy[:, 1] >= 0, :]
    xy = xy[xy[:, 1] < xrange.size, :]
    xy = np.concatenate((mu[np.newaxis, :], xy), axis=0)
    xy += 0.5
    """
    u, sv = proc_data['pupil'][0]['axdir'][frame_idx], proc_data['pupil'][0]['axlen'][frame_idx]
    n = 100  # Number of points around ellipse
    p = np.linspace(0, 2 * np.pi, n)[:, np.newaxis]
    xy = np.concatenate((np.cos(p), np.sin(p)), axis=1) * (sv**0.5) @ u
    xy += mu
    pupil_center = np.concatenate((mu[np.newaxis, :], xy), axis=0)

    # Plot the image
    img = cv2.convertScaleAbs(img, alpha=3.5, beta=1.5)
    ax.imshow(img, cmap='gray')
    ax.scatter(xy[:, 1], xy[:, 0], c='olive', s=.7, marker='+')
    ax.scatter(mu[1], mu[0], c='green', s=50, marker='+')
    ax.axis('off')
    ax.set_title("Eye view", fontsize=LABEL_FONTSIZE)
    # Plot a border around eye view image
    rect = plt.Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1, linewidth=1.5, edgecolor='green', facecolor='none')
    ax.add_patch(rect)

    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.02, pos[1]-.02, pos[2]*1.3, pos[3]*1.3])

    ## TODO: Add reflector to correct ellipse computation

def plot_network(ax, filepath):
    # Load the image
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    # Convert to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    ax.imshow(img_rgb)
    ax.axis('off')
    ax.text(0, 1.02, "Behavior to neural activity model", transform=ax.transAxes, ha="left")
            # medium fontsize 

    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.07, pos[1]-.01, pos[2]*1.6, pos[3]*1])

def plot_exp_illustration(ax, filepath):
    # Load the image
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    # Convert to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    ax.imshow(img_rgb)
    ax.axis('off')
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.01, pos[1]-.02, pos[2]*1.5, pos[3]*1.5])

def plot_recording_regions(ax):
    root = "/home/asyeda/pachitariulab_dm11/data/PROC/TX140/2024_11_06/1/" #LZ14/2024_06_20/1/"
    suite2p_ops = "/home/asyeda/pachitariulab_dm11/data/PROC/TX140/2024_11_06/1/suite2p/plane0/ops.npy"
    data = np.load(suite2p_ops, allow_pickle=True).item()

    if os.path.exists(f'{BASE_FOLDER}paper_figures/recording_regions_canvas.npy'):
        # Load the precomputed canvas
        canvas_data = np.load(f'{BASE_FOLDER}paper_figures/recording_regions_canvas.npy', allow_pickle=True).item()
        canvas = canvas_data['canvas']
        max_width = canvas_data['max_width']
        max_height = canvas_data['max_height']
    else:
        # First pass to get all max_proj images and determine canvas size
        max_projs = []
        max_width = 0
        max_height = 0

        print("Loading max_proj images and determining canvas size...")
        for n in tqdm(range(data["nplanes"])):
            ops = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "ops.npy"),
                allow_pickle=True,
            ).item()
            max_proj = ops['max_proj']
            dy = ops['dy']
            dx = ops['dx']
            print(ops['Lx'], ops['Ly'])
            
            max_projs.append((max_proj, dy, dx))
            
            # Calculate potential right and bottom edges
            right_edge = dx + max_proj.shape[1]
            bottom_edge = dy + max_proj.shape[0]
            
            if right_edge > max_width:
                max_width = right_edge
            if bottom_edge > max_height:
                max_height = bottom_edge

        # Create empty canvas
        canvas = np.ones((max_height, max_width), dtype=np.float32) * np.nan

        # Second pass to place each max_proj on the canvas
        for max_proj, dy, dx in (max_projs):
            h, w = max_proj.shape
            canvas[dy:dy+h, dx:dx+w] = max_proj

        # save canvas to file so you don't have to recompute
        np.save(f'{BASE_FOLDER}paper_figures/recording_regions_canvas.npy', {'canvas': canvas, 'max_width': max_width, 'max_height': max_height})
        
    # Display the result
    rotated_canvas = np.rot90(canvas, k=1)  # rotate 90° counterclockwise
    flip_canvas = np.fliplr(rotated_canvas)  # flip left-right
    ax.imshow(flip_canvas, cmap='gray', vmax=7500)
    ax.axis('off')
    ax.text(0.5, 1.05, "2-photon imaging\nregions", transform=ax.transAxes, ha="center")
    ax.set_aspect(.5/.75) #canvas.shape[1]/canvas.shape[0]) # .75/.5
    # Add scalebar to the plot
    scalebar_physical_um = 500  # Desired scale bar size in micrometers
    scalebar_pixel_width = 250 #scalebar_physical_um / pixel_to_um_ratio  # Convert to pixels

    # Set the start and end points of the scale bar    
    scale_bar_start = 1575 #ax.get_xlim()[0]
    scale_bar_end = scale_bar_start + scalebar_pixel_width
    ax.plot([scale_bar_start, scale_bar_end], [ax.get_ylim()[1]+150, ax.get_ylim()[1]+150], 'k', 
            linestyle='-', lw=2, clip_on=False, markeredgecolor='k')
    ax.text(scale_bar_start + scalebar_pixel_width/2, ax.get_ylim()[1]+140, f'{int(scalebar_physical_um)} $\mu$m', ha='center', 
            va='bottom', fontsize=LABEL_FONTSIZE-3, color='k')
    
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+.015, pos[1]-.025, pos[2]*1.3, pos[3]*1.3])

    # Add brain region overlay
    brain_region_img = f'{BASE_FOLDER}paper_figures/brain_recording_region.png'
    import matplotlib.image as mpimg
    brain_region = mpimg.imread(brain_region_img)
    #brain_region_gray = cv2.cvtColor(brain_region, cv2.COLOR_BGR2GRAY)
    axis_inset = ax.inset_axes([-.2, .8, .35, .35])
    # pos to top left
    axis_inset.axis('off')  # Hide the inset axis
    axis_inset.imshow(brain_region)


def scatter_plot(fig, gs, recording_type):
    
    if recording_type == 'static':
        data = np.load(f"{BASE_FOLDER}nonlinear_lightstatic_results_udcnv.npy", allow_pickle=True).item()
    else:
        data = np.load(f"{BASE_FOLDER}nonlinear_lightrandom_results_udcnv.npy", allow_pickle=True).item()
    
    ax = plt.subplot(gs)
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+.04, pos[1]+.025, pos[2]*1.25, pos[3]*1.25])
    axin = ax.inset_axes([0.6, 0.6, 0.4, 0.4])

    num_animals = len(data)
    mean_pupil_list = []
    mean_face_list = []
    for i in range(num_animals):
        # Skip TX150 because it has a different sampling rate
        if "TX150" in list(data.keys())[i]:
            continue
        mname = list(data.keys())[i]

        if recording_type == 'static':
            select_mname = 'TX140'
        else:
            select_mname = 'TX144'

        pupil_ve = data[mname]['pupil']["varexp_neurons_before_carbachol"]*100
        face_ve = data[mname]['face_eye']["varexp_neurons_before_carbachol"]*100
        if mname == select_mname:
            ax.scatter(pupil_ve, face_ve, s=.4, color=rand_stat_colors[rand_stat_color_names.index(recording_type)], alpha=0.4, edgecolor='none')

        ax.set_xlabel("Pupil position", color=pupil_label_color)
        if recording_type == 'static':
            ax.set_ylabel("Face w/o eye", color=face_eye_label_color)
            ax.text(-.21, 1.08, "Single neuron variance explained (%)", transform=ax.transAxes, ha="left", fontsize=LABEL_FONTSIZE-2)
        ax.axis('square')
        ax.set_xlim(-10, 62)
        ax.set_ylim(-10, 62)
        ax.plot([-100, 40], [-100, 40], 'k--')
        # Set x and y ticks at 0, 30, 60
        ax.set_xticks([0, 30, 60])
        ax.set_yticks([0, 30, 60])

        # Compute average for each mouse (face w/o eye vs pupil position)
        mean_pupil_ve = np.nanmean(data[mname]['pupil']["varexp_neurons_before_carbachol"]*100)
        mean_face_ve = np.nanmean(data[mname]['face_eye']["varexp_neurons_before_carbachol"]*100)
        mean_pupil_list.append(mean_pupil_ve)
        mean_face_list.append(mean_face_ve)

        # plot an inset showing mean values
        axin.scatter(mean_pupil_ve, mean_face_ve, s=1.5, color=rand_stat_colors[rand_stat_color_names.index(recording_type)], alpha=1, 
                         edgecolors=rand_stat_colors[rand_stat_color_names.index(recording_type)], marker='x')   
        axin.plot([0,7], [0,7], 'k--', lw=0.5)
        axin.set_xlim(0, 7)
        axin.set_ylim(0, 7)
        axin.axis('square')
        axin.set_xticks([0, 5])
        axin.set_yticks([0, 5])

    print("Single neuron recording type: ", recording_type)
    print("Mean pupil position VE across mice: ", np.mean(mean_pupil_list))
    print("Mean face w/o eye VE across mice: ", np.mean(mean_face_list))

    # Paired t-test across mice
    t_stat, p_val = ttest_rel(mean_face_list, mean_pupil_list)
    print("single neuron", recording_type, p_val)
    # Annotate the t-test result on inset
    if p_val < 0.05:
        axin.text(3, 6, '*', fontsize=10)
    elif p_val < 0.01:
        axin.text(3, 6, '**', fontsize=10)
    elif p_val < 0.001:
        axin.text(3, 6, '***', fontsize=10)
    else:
        axin.text(3, 8, 'n.s.', fontsize=9)

def line_plot(fig, gs, recording_type):
    if recording_type == 'static':
        nonlinear_results = np.load(f"{BASE_FOLDER}nonlinear_lightstatic_results_udcnv.npy", allow_pickle=True).item()
    else:
        nonlinear_results = np.load(f"{BASE_FOLDER}nonlinear_lightrandom_results_udcnv.npy", allow_pickle=True).item()
    
    # remove TX150 from nonlinear_results
    nonlinear_results = {k: v for k, v in nonlinear_results.items() if "TX150" not in k}

    # Plot configuration
    ax = plt.subplot(gs)
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+.04, pos[1]+.08, pos[2], pos[3]*0.75])

    # Plot line plot
    input_labels = ['Pupil area', 'Pupil position', 'Face w/o eye', 'Face w/o eye\n+ pupil position']
    alpha = 0.05  # significance level
    alpha2 = 0.001  # significance level
    varexp_data = []  # Collect normalized variance for all animals

    # Collect normalized variance explained for all animals
    for mname in nonlinear_results:
        varexp_norms = []
        for name in ['pupil_area', 'pupil', 'face_eye', 'face_eye_pupil']:
            value = (nonlinear_results[mname][name]['vexp_before_carbachol']) * 100 # vexp_after_carbachol vexp_before_carbachol
            varexp_norms.append(value)
            #print(f"{recording_type} {name} {value}")
        varexp_data.append(varexp_norms)

    # Convert to NumPy array for easier manipulation
    varexp_data = np.array(varexp_data)

    # Remove the singleton dimension
    varexp_data = np.squeeze(varexp_data)  # Shape becomes (4, 4)
    print(">>>>>>> recording type: ", recording_type)
    # Pairwise comparisons 
    pairwise_pvals = []
    for i in range(len(input_labels)):
        for j in range(i + 1, len(input_labels)):
            _, pval = ttest_rel(varexp_data[:, i], varexp_data[:, j])
            #_, pval = wilcoxon(varexp_data[:, i], varexp_data[:, j])
            pairwise_pvals.append((i, j, pval))

    # print results for paper
    means = varexp_data.mean(axis=0)
    stds = varexp_data.std(axis=0, ddof=1)

    print("=== Mean ± SD of normalized variance explained (%): ===")
    for label, m, s in zip(input_labels, means, stds):
        print(f"{label:15s}: {m:5.1f}\% $\pm$ {s:4.1f}\%")

    # Plot results
    for i, animal_data in enumerate(varexp_data):
        mouse_name = list(nonlinear_results.keys())[i]
        ani_color = rand_stat_colors[1] if recording_type == 'static' else rand_stat_colors[0]
        #ani_color = 'red' if mouse_name in ['FXa8', 'FXa10'] else ani_color
        ax.plot(animal_data, '-', markersize=3, lw=.9, color=ani_color, alpha=.8,
                    label=f"Mouse {list(nonlinear_results.keys())[i]}")
    # Plot average line across all mice
    ax.plot(np.mean(varexp_data, axis=0), '-', label='Average', markersize=3, color='black', lw=2)
    #ax.legend()
    ax.set_xticks(np.arange(len(input_labels)), input_labels, rotation=45, ha='right', rotation_mode='anchor')
    # change color of x ticks 'top', 'bottom', 'center', 'baseline', 'center_baseline'
    xtick_colors = [pupil_area_label_color, pupil_label_color, face_eye_label_color, face_eye_pupil_label_color]
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(xtick_colors[i])

    if recording_type == 'static':
        ax.set_ylabel("Neural PC\nvariance explained (%)")
    # Add inset axis for significance annotations above the plot
    axin = ax.inset_axes([0, 1.05, 1, 0.3])  # Position the inset axis above the main axis
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
                    y = prev_line_y + 2
                axin.plot([i, j], [y, y], 'k-', lw=.7)  # Use lists for plotting horizontal line
                if corrected_pval[2] < alpha2:
                    axin.text((i + j) / 2, y-1.7, '***', ha='center')
                else:
                    axin.text((i + j) / 2, y-1.6, '*', ha='center')
                prev_line_y = y  # Update previous line y-coordinate
            else: # add n.s
                if prev_line_y == 0:
                    y = varexp_data[:, [i, j]].max()
                else:
                    y = prev_line_y + 1.5
                axin.plot([i, j], [y, y], 'k-', lw=.7)  # Use lists for plotting horizontal line
                axin.text((i + j) / 2, y + 0.02, 'n.s.', ha='center', fontsize=8)
                prev_line_y = y

    ax.set_ylim(0, 40)
    ax.set_yticks([0, 20, 40])

def stim_presentation(gs, filepath):
    nested_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs, hspace=.6)
    
    # Static image
    ax = plt.subplot(nested_gs[0, 0])
    # Load the image
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax.imshow(img_rgb)
    ax.axis('off')
    ax.set_title("Static image", color=rand_stat_colors[1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.015, pos[1]+.04, pos[2]*1.2, pos[3]*1.3])
    # add text at the bottom of the plot saying full recording
    ax.text(0.5, -0.1, "1 image; 1 hr recording", ha='center', va='top', 
            transform=ax.transAxes, fontsize=LABEL_FONTSIZE-2, color='black')

    # Load your images (replace with your image paths)
    image_paths = [f'{BASE_FOLDER}/images/circles_test.jpg',
                f'{BASE_FOLDER}/images/dryland_test.jpg',
                f'{BASE_FOLDER}/images/leaves_test.jpg',
                f'{BASE_FOLDER}/images/paved_test.jpg',
                f'{BASE_FOLDER}/images/rleaves_test.jpg']
    images = [cv2.imread(path) for path in image_paths]

    # Ensure all images have an alpha channel (RGBA)
    for i, img in enumerate(images):
        if img.shape[2] == 3:
            images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Image spacing and offset
    x_spacing = 15
    y_spacing = 35
    x_offset, y_offset = 0, 0  # Start at top-left

    # Image dimensions
    img_width = max(img.shape[1] for img in images)
    img_height = max(img.shape[0] for img in images)

    # Calculate canvas size: each image + grey per image → 2 * images
    num_slots = 2 * len(images)
    total_width = x_offset + img_width + (num_slots - 1) * x_spacing
    total_height = y_offset + img_height + (num_slots - 1) * y_spacing

    # Create final canvas
    canvas = np.zeros((total_height, total_width, 4), dtype=np.uint8)

    # Create a single grey image with alpha
    grey = np.ones((img_height, img_width, 4), dtype=np.uint8) * 128
    grey[:, :, 3] = 255  # Fully opaque

    # Interleave grey and image
    for i, img in enumerate(images):
        # Draw grey background
        gx, gy = x_offset, y_offset
        canvas[gy:gy+img_height, gx:gx+img_width] = grey

        # Update offset for actual image
        x_offset += x_spacing
        y_offset += y_spacing

        # Draw actual image
        ix, iy = x_offset, y_offset
        canvas[iy:iy+img.shape[0], ix:ix+img.shape[1]] = img

        # Advance offset again for next grey slot
        x_offset += x_spacing
        y_offset += y_spacing

    # Random images
    ax = plt.subplot(nested_gs[1, 0])
    ax.imshow(canvas)
    ax.axis('off')  # Hide axes
    ax.set_title("Random images", color=rand_stat_colors[0])

    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.015, pos[1]+.04, pos[2]*1.2, pos[3]*1.3])
    ax.text(0.5, -0.1, "~5000 images (3Hz);\n1-2 hr recording", ha='center', va='top', 
            transform=ax.transAxes, fontsize=LABEL_FONTSIZE-2, color='black')

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

def get_behavioral_data(recording_type):

    if recording_type == 'static':
        dbs = pdh.get_light_static_info() 
        base_folder = f'{BASE_FOLDER}light_recordings_static/'
        mname = 'FXa8'
    else:
        dbs = pdh.get_light_random_info() 
        base_folder = f'{BASE_FOLDER}light_recordings_random/'
        mname = 'FXa8' 

    datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]
    # Load blink data
    blink_data = np.load(f'{BASE_FOLDER}blink_times.npy', allow_pickle=True).item()        
    blink_start_idx = blink_data['light_{}'.format(recording_type)][mname]['1'][:,-2].astype(int)
    blink_end_idx = blink_data['light_{}'.format(recording_type)][mname]['1'][:,-1].astype(int)
    blink_idx = []
    for s, e in zip(blink_start_idx, blink_end_idx):
        if e > s:
            blink_idx = np.append(blink_idx, np.arange(s-5, e+6)) # few frames before and after blink removed
        else:
            blink_idx = np.append(blink_idx, s)
    blink_idx = np.unique(blink_idx.astype(int))

    # Check if file exists 
    if os.path.exists(f"{base_folder}/{mname}_{datexp}_rasterdata_udcnv.npz"):
        data = np.load(f"{base_folder}/{mname}_{datexp}_rasterdata_udcnv.npz", allow_pickle=True)
        X_embedding = data['X_embedding']
        aligned_saccades = data['aligned_saccades']
        pupil_x = data['pupil_x']
        pupil_y = data['pupil_y']
        exact_saccades = data['exact_saccades']
        pupil_timestamp = data['pupil_timestamp']
        spks_timestamp = data['spks_timestamp']
        motSVD = data['motSVD']
    else:
        # Load saccade data
        saccade_data = np.load(f'{BASE_FOLDER}saccade_times.npy', allow_pickle=True).item()
        _, _, _, _, _, _, _, npz_data = nonlinear_modelv3.load_data(base_folder, mname, datexp)
        
        # Load spks using the new deconvolution method (udcnv)
        spks_udcnv_file = "/media/Atika/neural_data_udcnv/{}_{}_1_spks.npz".format(mname, datexp)
        spks_udcnv = np.load(spks_udcnv_file)['spks']
        
        spks = []
        for block_idx in range(1): # CHANGE THIS IDX
            block = list(npz_data['tcam'].item().keys())[block_idx]
            # Resample behavior data to neural timestamps
            spks.append(helper_functions.segment_data(spks_udcnv.T, npz_data['neural_frames_list'], block_idx).T)
        spks = np.concatenate(spks, axis=1)

        pupil_x = npz_data['pupil_x_polar']#npz_data['pupil_x_small']
        pupil_y = npz_data['pupil_y_polar'] #npz_data['pupil_y_small']

        blocks = list(npz_data['tcam'].item())
        blk = blocks[0].split('blk')[1]
        
        pupil_timestamp = npz_data['tcam'].item()['blk1']
        spks_timestamp = npz_data['tneural'].item()['blk1']

        # Filter out high-frequency noise in pupil data
        tmin = np.min(pupil_timestamp)
        cam_time = (pupil_timestamp - tmin) * 24 * 60 * 60
        sampling_rate = (1 / np.mean(np.diff(cam_time))).astype(int)

        exact_saccades = saccade_data["light_"+recording_type][mname][blk] # Get exact saccade indices from precomputed data
        
        model = Rastermap().fit(zscore(spks.copy(), axis=1))
        y = model.embedding # neurons x 1
        isort = model.isort
        X_embedding = zscore(utils.bin1d(spks[isort], bin_size=200, axis=0), axis=1)
        
        neural_saccade_times, aligned_saccades = get_aligned_saccade_times(exact_saccades, spks_timestamp)
        aligned_saccades = aligned_saccades.astype(int)

        # Save data to npz file 
        motSVD = npz_data['face_svd'].item()['motSVD'][0]
        data = {"X_embedding": X_embedding, "aligned_saccades": aligned_saccades, "pupil_x": pupil_x, "pupil_y": pupil_y, 
                "pupil_timestamp": pupil_timestamp, "exact_saccades": exact_saccades, "neural_saccade_times": neural_saccade_times,
                  "spks_timestamp": spks_timestamp, "motSVD": motSVD, "sampling_rate": sampling_rate, "isort": isort}  
        np.savez(f"{base_folder}/{mname}_{datexp}_rasterdata_udcnv.npz", **data)

    return X_embedding, aligned_saccades, pupil_x, pupil_y, exact_saccades, pupil_timestamp, spks_timestamp, motSVD, blink_idx, mname

"""
def raster_plot(fig, gs, recording_type):

    # Get stim ID data
    stim_id_path = "/media/Atika/tuning_curve_data/stim_ID_lightstaticrandom_recordings/"
    stim_id_file = glob(f"{stim_id_path}/*{recording_type}*.mat")[0]
    stimID = loadmat(stim_id_file)['TL']['stim'][0,0]['istim'][0,0].squeeze().flatten()

    X_embedding, aligned_saccades, pupil_x, pupil_y, exact_saccades, pupil_timestamp, spks_timestamp, motSVD, blink_idx, mname = get_behavioral_data(recording_type)
    print("X_embedding before: ", X_embedding.shape, "STIM ID shape: ", stimID.shape, "spks_timestamp shape: ", spks_timestamp.shape)
    # Light recordings truncated for model training to match session length
    pred_data = np.load("/home/asyeda/pupil_analysis/light_recordings_{}/rastermap_out_udcnv.npy".format(recording_type), allow_pickle=True).item()
    neural_start_idx = pred_data[mname]['face_eye']['tneural_start']
    neural_end_idx = pred_data[mname]['face_eye']['tneural_end']
    X_embedding = X_embedding[:, neural_start_idx:neural_end_idx]
    spks_timestamp = spks_timestamp[neural_start_idx:neural_end_idx]
    stimID = stimID[neural_start_idx:neural_end_idx]

    print("X_embedding after: ", X_embedding.shape, "STIM ID shape: ", stimID.shape, "spks_timestamp shape: ", spks_timestamp.shape)
    neural_xmin = 2500 #0
    neural_xmax = 3200 #4000
    pupil_xmin = np.argmin(np.abs(pupil_timestamp - spks_timestamp[neural_xmin]))
    pupil_xmax = np.argmin(np.abs(pupil_timestamp - spks_timestamp[neural_xmax]))

    # Nan out values during blink times
    pupil_x_nan = pupil_x.copy()
    pupil_y_nan = pupil_y.copy()
    pupil_x_nan[blink_idx] = np.nan
    pupil_y_nan[blink_idx] = np.nan

    vline_width = .3

    nested_gs = GridSpecFromSubplotSpec(4, 1, height_ratios=[.7, .05, .2, .1], subplot_spec=gs, hspace=0.03)

    neural_xlim = [neural_xmin, neural_xmax]
    pupil_xlim = [pupil_xmin, pupil_xmax]

    plot_x_adjust = 0.04
    # Plot rastermap sorted by pupil area correlation
    ax0 = plt.subplot(nested_gs[0])
    raster = ax0.imshow(X_embedding, vmin=0, vmax=1, cmap="gray_r", aspect="auto")
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.set_ylim(X_embedding.shape[0],0)
    ax0.set_xlim(neural_xlim)
    #for s in aligned_saccades:
    #    ax0.axvline(s, color=(0,1,0), linestyle='dashed', lw=vline_width)
    remove_spines(ax0)
    # Add y-axis scale bar for rastermap
    bin_size = 100
    scale_bar_length = 50
    scale_bar_start = ax0.get_ylim()[0] - 50
    scale_bar_end = scale_bar_start + scale_bar_length
    move_x = -4
    ax0.plot([ax0.get_xlim()[0]+move_x, ax0.get_xlim()[0]+move_x], [scale_bar_start, scale_bar_end], 'k-', lw=1.5, clip_on=False)
    if recording_type == 'static':
        ax0.text(ax0.get_xlim()[0]+(move_x*5), scale_bar_start - scale_bar_length - 20, 
             f'{bin_size*scale_bar_length} neurons', ha='left', va='center', rotation='vertical', fontsize=LABEL_FONTSIZE-3)
    else:
        ax0.text(ax0.get_xlim()[0]+(move_x*5), scale_bar_start - scale_bar_length, 
            f'{bin_size*scale_bar_length} neurons', ha='left', va='center', rotation='vertical', fontsize=LABEL_FONTSIZE-3)
       
    # shift plot to the right 
    pos = ax0.get_position().bounds
    ax0.set_position([pos[0]+plot_x_adjust, pos[1], pos[2], pos[3]])

    if recording_type == 'random':
        # Add x-axis scale bar for pupil area, x and y, rastermap on the right
        duration = 10 # seconds
        sampling_rate = 3 #Hz
        scale_bar_length = sampling_rate * duration
        scale_bar_start = ax0.get_xlim()[1] - 40
        scale_bar_end = scale_bar_start + scale_bar_length
        below_y = 180
        ax0.plot([scale_bar_start, scale_bar_end], [ax0.get_ylim()[0]+below_y, ax0.get_ylim()[0]+below_y], 'k-', lw=1.5, clip_on=False)
        ax0.text(scale_bar_start + scale_bar_length/2, ax0.get_ylim()[0]+below_y, f'{int(duration)} s'.format(str(duration)), 
                 ha='center', va='bottom', fontsize=LABEL_FONTSIZE-3)

    # Add a title to the top of the plot
    if recording_type == 'random':
        ax0.text((ax0.get_xlim()[0]+ax0.get_xlim()[1])//2, ax0.get_ylim()[1], f"{recording_type.capitalize()} images recording (new image every ~0.33s)",
            fontsize=LABEL_FONTSIZE, color=rand_stat_colors[rand_stat_color_names.index(recording_type)],
            ha='center', va='bottom', clip_on=False)
    else:
        ax0.text((ax0.get_xlim()[0]+ax0.get_xlim()[1])//2, ax0.get_ylim()[1], f"{recording_type.capitalize()} image recording (same image for one hour)",
            fontsize=LABEL_FONTSIZE, color=rand_stat_colors[rand_stat_color_names.index(recording_type)],
            ha='center', va='bottom', clip_on=False)
    
    if recording_type == 'static':
        # Get the current position of the axis (ax) in figure coordinates
        pos = ax0.get_position()  # [x0, y0, width, height]

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

    # Plot stim ID
    ax = plt.subplot(nested_gs[1])
    # Get RGB colors for each stimID
    colors = np.array([id_to_color(s) for s in stimID])
    ax.imshow(colors[np.newaxis, :, :], aspect='auto')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(neural_xlim)
    # Add vertical text on the left without moving axes
    ax.text(
        x=ax.get_xlim()[0] - 95,  # slightly left of first data point
        y=2.5,                        # vertical position (since image is 1 pixel high)
        s="Stimulus ID",
        fontsize=LABEL_FONTSIZE-4,
        ha='left',                 # right-align text so it hugs the plot
        va='bottom',                 # align with bottom of the horizontal image
        clip_on=False,                # allow text to go outside axes limits,
    )
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+plot_x_adjust, pos[1], pos[2], pos[3]*.4])
    remove_spines(ax)

    # Plot pupil x and y
    pupil_x_color = 'darkgreen'
    pupil_y_color = 'lightgreen'
    ax1 = plt.subplot(nested_gs[2])
    ax1.plot(pupil_x_nan, color=pupil_x_color, label='Azimuth', lw=.5)
    ax1.plot(pupil_y_nan, color=pupil_y_color, label='Elevation', lw=.5)

    ax1.set_xlim(pupil_xlim)
    ax1.set_yticks([])
    ax1.set_xticks([])
    remove_spines(ax1)
     # Add y-axis scale bar for pupil x and y
    scale_bar_length = 5
    scale_bar_start = ax1.get_ylim()[1] - 14
    scale_bar_end = scale_bar_start + scale_bar_length
    move_x = + 65
    ax1.plot([ax1.get_xlim()[1]+move_x, ax1.get_xlim()[1]+move_x], [scale_bar_start-23, scale_bar_end-23], 'k-', lw=1.5, clip_on=False)
    ax1.text(ax1.get_xlim()[1]+(move_x*2), scale_bar_start + scale_bar_length/2-21, f'{scale_bar_length}$^\circ$',
              fontsize=LABEL_FONTSIZE-3)
    if recording_type == 'random':
        x_adjust = 7
        y_adjust = -17
    else:
        x_adjust = 10.5
        y_adjust = -18
    move_x = -120 #-68
    ax1.text(ax1.get_xlim()[0]+(move_x*x_adjust), scale_bar_end-7, 'Elevation',
            fontsize=LABEL_FONTSIZE-4, color=pupil_y_color)
    ax1.text(ax1.get_xlim()[0]+(move_x*x_adjust), scale_bar_end+y_adjust-7, 'Azimuth',
            fontsize=LABEL_FONTSIZE-4, color=pupil_x_color)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+plot_x_adjust, pos[1]-0.01, pos[2], pos[3]])

    # Plot motSVD 
    ax2 = plt.subplot(nested_gs[3])
    ax2.plot(motSVD[:,0], c='orange', lw=.35, label='motion PC')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim(pupil_xlim)
    #for s in exact_saccades:
    #    ax2.axvline(s[0].astype(int), color=(0,1,0), linestyle='dashed', linewidth=vline_width)
    remove_spines(ax2)
    ax2.text(ax2.get_xlim()[0]+(move_x*x_adjust), scale_bar_end+y_adjust, 'motion PC',
            fontsize=LABEL_FONTSIZE-4, color='orange')
    ax2.set_ylim(np.min(motSVD[pupil_xlim[0]:pupil_xlim[1],0]), np.max(motSVD[pupil_xlim[0]:pupil_xlim[1],0]))
    # shift plot to the right 
    pos = ax2.get_position().bounds
    ax2.set_position([pos[0]+plot_x_adjust, pos[1], pos[2], pos[3]])

    return fig
"""

import hashlib

def id_to_color(stim_id):
    # Convert stim_id to a reproducible RGB color using hashing
    h = hashlib.md5(str(stim_id).encode()).hexdigest()
    rgb = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    return rgb


def raster_plot_light_predictions(fig, gs, recording_type):
    
    nested_gs = GridSpecFromSubplotSpec(5, 1, height_ratios=[.35, .35, .1, .15, .1], subplot_spec=gs, hspace=0.03)

    # Get stim ID data
    stim_id_path = "/media/Atika/tuning_curve_data/stim_ID_lightstaticrandom_recordings/"
    stim_id_file = glob(f"{stim_id_path}/*{recording_type}*.mat")[0]
    stimID = loadmat(stim_id_file)['TL']['stim'][0,0]['istim'][0,0].squeeze().flatten()

    if recording_type == 'static':
        dbs = pdh.get_light_static_info() 
        mname = 'TX140'
    else:
        dbs = pdh.get_light_random_info() 
        mname = 'TX144' 
    datexp = [db['datexp'] for db in dbs if db['mname'] == mname][0]
    
    # Load neural and pupil data
    xlim_neural = [2600, 3000] # 50 
    n_folds = 5
    vline_width = .4
    if mname in ['FXa10', 'FXa8', 'FXa9', 'FXa1']:
        spks, _, _, _, _, _, _, npz_data = nonlinear_modelv3.load_data(f'/home/asyeda/nas113/Atika/{mname}/{datexp}/{1}', mname, datexp) # different base folder for these mice
    else:
        _, _, _, _, _, _, _, npz_data = nonlinear_modelv3.load_data(os.path.join(BASE_FOLDER, f'light_recordings_{recording_type}'), mname, datexp)
        spks_udcnv_file = "/media/Atika/neural_data_udcnv/{}_{}_1_spks.npz".format(mname, datexp)
        spks = np.load(spks_udcnv_file)['spks']

    if not os.path.exists(f"/media/Atika/eye_movement_paper/pupil_analysis/rasterdata/{mname}_{datexp}_fig1rasterdata.npz"):
        rastermap_out = np.load(f"{BASE_FOLDER}light{recording_type}_rastermap_out_udcnv.npy".format(recording_type), allow_pickle=True).item()
        # Get isort if saved 
        if os.path.exists(f"{BASE_FOLDER}rasterdata/{mname}_{datexp}_rasterdata_udcnv.npz"):     # check if isort saved
            data = np.load(f"{BASE_FOLDER}rasterdata/{mname}_{datexp}_rasterdata_udcnv.npz", allow_pickle=True)
            isort = data['isort']
        else: # Run rastermap 
            # Load spks using the new deconvolution method (udcnv)
            if mname in ['FXa10', 'FXa8', 'FXa9', 'FXa1']:
                spks, _, _, _, _, _, _, npz_data = nonlinear_modelv3.load_data(f'/home/asyeda/nas113/Atika/{mname}/{datexp}/{1}', mname, datexp) # different base folder for these mice
            else:
                spks_udcnv_file = "/media/Atika/neural_data_udcnv/{}_{}_1_spks.npz".format(mname, datexp)
                spks = np.load(spks_udcnv_file)['spks']
            model = Rastermap().fit(zscore(spks, axis=1).copy())
            isort = model.isort
            np.savez(f"{BASE_FOLDER}rasterdata/{mname}_{datexp}_rasterdata_udcnv.npz", isort=isort)

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

        np.savez(f"/media/Atika/eye_movement_paper/pupil_analysis/rasterdata/{mname}_{datexp}_fig1rasterdata.npz", y_test_embedding=y_test_embedding, 
                y_pred_pupil_embedding=y_pred_pupil_embedding, y_pred_face_embedding=y_pred_face_embedding,
                itest_sorting=itest_sorting, x_test_pupil=x_test_pupil, x_test_motsvd=x_test_motsvd,
                tstart=tstart, tend=tend)
    else:
        data = np.load(f"/media/Atika/eye_movement_paper/pupil_analysis/rasterdata/{mname}_{datexp}_fig1rasterdata.npz", allow_pickle=True)
        y_test_embedding = data['y_test_embedding']
        y_pred_pupil_embedding = data['y_pred_pupil_embedding']
        itest_sorting = data['itest_sorting']
        x_test_pupil = data['x_test_pupil']
        x_test_motsvd = data['x_test_motsvd']
        tstart = data['tstart'].item()
        tend = data['tend'].item()
    

    stimID = stimID[tstart:tend]
    saccade_data = np.load(f'{BASE_FOLDER}saccade_times.npy', allow_pickle=True).item()
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
    
    # Colormap with transparent NaNs
    cmap = plt.cm.gray_r.copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    vmin, vmax = 0, 1.5

    # First subplot (Raw)
    ax1 = plt.subplot(nested_gs[0])
    raster = ax1.imshow(y_test_embedding[:,itest_sorting], aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    if recording_type == 'static':
        ax1.set_title("Static image", color=rand_stat_colors[1])
    else:
        ax1.set_title("Random images", color=rand_stat_colors[0])
    #ax1.set_title('{}'.format(recording_type), color=rand_stat_colors[rand_stat_color_names.index(recording_type)], fontsize=LABEL_FONTSIZE)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(xlim_neural)
    helper_functions.remove_spines(ax1)  
    # Add y-axis scale bar for rastermap
    bin_size = 200
    scale_bar_length = 25
    scale_bar_start = ax1.get_ylim()[0] - 50
    scale_bar_end = scale_bar_start + scale_bar_length
    # orient the text vertically
    ax1.plot([ax1.get_xlim()[0]-(0.01*np.diff(ax1.get_xlim())), ax1.get_xlim()[0]-(0.01*np.diff(ax1.get_xlim()))],
              [scale_bar_start, scale_bar_end], 'k-', lw=2, clip_on=False)
    ax1.text(ax1.get_xlim()[0]-(0.05*np.diff(ax1.get_xlim())), scale_bar_start-50, f'{scale_bar_length*bin_size} neurons',
              ha='left', va='center', rotation='vertical', fontsize=LABEL_FONTSIZE-3)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0], pos[1]+.01, pos[2], pos[3]])
    # Add x-axis scale bar for pupil area, x and y, rastermap on the right
    duration = 10 # seconds
    scale_bar_length = 3 * duration
    scale_bar_start = ax1.get_xlim()[1] - 70
    scale_bar_end = scale_bar_start + scale_bar_length
    y_pos = ax1.get_ylim()[1]-5
    ax1.plot([scale_bar_start, scale_bar_end], [y_pos, y_pos], 'k-', lw=1, clip_on=False)
    ax1.text(scale_bar_start + scale_bar_length/2, y_pos, f'{int(duration)} s'.format(str(duration)), 
             ha='center', va='bottom', clip_on=False, fontsize=LABEL_FONTSIZE-2)
    
    #  subplot (Predicted)
    ax2 = plt.subplot(nested_gs[1])
    ax2.imshow(y_pred_pupil_embedding[:,itest_sorting], aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_title('Prediction from pupil position', color='darkgreen', fontsize=LABEL_FONTSIZE-2)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlim(xlim_neural)
    helper_functions.remove_spines(ax2)  
    pos = ax2.get_position().bounds
    ax2.set_position([pos[0], pos[1]-.03, pos[2], pos[3]])

    # Plot stim ID
    ax3 = plt.subplot(nested_gs[2]) # plot stim ID  
    colors = np.array([id_to_color(s) for s in stimID])  # Get RGB colors for each stimID
    ax3.imshow(colors[np.newaxis, :, :], aspect='auto')
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_xlim(xlim_neural)
    ax3.set_title("Stimulus ID", fontsize=LABEL_FONTSIZE-4, pad=-1)
    pos = ax3.get_position().bounds
    ax3.set_position([pos[0], pos[1]-.01, pos[2], pos[3]*.15])
    remove_spines(ax3)    
    
    #  subplot (Pupil X and Y)
    ax4 = plt.subplot(nested_gs[3])
    ax4.plot(x_test_pupil[:,0][itest_sorting], label='Azimuth', lw=.5, color=pupil_x_color)
    ax4.plot(x_test_pupil[:,1][itest_sorting], label='Elevation', lw=.5, color=pupil_y_color)
    ax4.set_xlim(xlim_neural)
    ax4.set_xticks([])
    ax4.margins(x=0)   
    helper_functions.remove_spines(ax4)  
    ax4.set_yticks([])
    legend = ax4.legend(
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(-.1, 1.15),
        bbox_transform=ax4.transAxes,
        frameon=False,
        fontsize=LABEL_FONTSIZE - 2,
        handlelength=0
    )

    legend.get_texts()[0].set_color(pupil_x_color)
    legend.get_texts()[1].set_color(pupil_y_color)

    # Add y-axis scale bar for pupil x and y
    scale_bar_length = 10
    scale_bar_start = ax4.get_ylim()[0] + 25
    scale_bar_end = scale_bar_start + scale_bar_length
    #ax4.plot([scale_bar_start, scale_bar_end], [scale_bar_start, scale_bar_end], 'k-', lw=1, clip_on=False, zorder=10)
    #ax4.text(scale_bar_start, scale_bar_start + scale_bar_length/2, f'{scale_bar_length}$^\circ$', ha='left', va='center', clip_on=False)
    pos = ax4.get_position().bounds
    ax4.set_position([pos[0], pos[1]-.01, pos[2], pos[3]])
    
    #  subplot svd 
    ax5 = plt.subplot(nested_gs[4])
    line, = ax5.plot(x_test_motsvd[:,0][itest_sorting], label='Motion SVD', color='orange', lw=1.2)
    legend=ax5.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.7), facecolor='none', fontsize=LABEL_FONTSIZE-2)
    legend.get_texts()[0].set_color('orange') 
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)    
    ax5.set_xlim(xlim_neural)
    ax5.margins(x=0)
    helper_functions.remove_spines(ax5)
    ax5.set_yticks([])
    ax5.set_xticks([])
    # adjust pos of ax5 
    pos = ax5.get_position().bounds
    ax5.set_position([pos[0], pos[1]-.01, pos[2], pos[3]*.75])
    
    """
    # add saccade lines to all plots
    for s in saccades_new:
        ax1.axvline(s, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
        ax2.axvline(s, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
        ax4.axvline(s, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
        ax5.axvline(s, color=(0,1,0), linestyle='dashed', linewidth=vline_width)
    """
    # set position of the entire plot
    for ax_idx in [ax1, ax2, ax3, ax4, ax5]:
        pos = ax_idx.get_position().bounds
        if recording_type == 'static':
            ax_idx.set_position([pos[0]+0.02, pos[1], pos[2]*1.05, pos[3]])
        else:
            ax_idx.set_position([pos[0]+0.01, pos[1], pos[2]*1.05, pos[3]])
            

    # Get the current position of the axis (ax) in figure coordinates
    pos = ax1.get_position()  # [x0, y0, width, height]
    if recording_type == 'static':
        cax = fig.add_axes([
        pos.x0+pos.width-pos.width*.2,           # align with left of main plot
        pos.y0 - 0.01,          # place slightly below the plot
        pos.width*.2,              # same width as main plot
        pos.height * 0.03       # thin horizontal bar
    ])
        cbar = plt.colorbar(raster, cax=cax, orientation="horizontal")
        cbar.set_label("z-scored", fontsize=LABEL_FONTSIZE - 3, labelpad=-.5)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.tick_params(labelsize=LABEL_FONTSIZE - 5)



def plot_train_test_tuning_scatter(gs):

    category = ['light_random', 'light_static']
    # Load train/test data if already computed
    if os.path.exists('/media/Atika/tuning_curve_data/'):
        category_test_data = np.load('/media/Atika/tuning_curve_data/tuning_test_data.npy', allow_pickle=True).item()
        category_train_data = np.load('/media/Atika/tuning_curve_data/tuning_train_data.npy', allow_pickle=True).item()
    else:
        print("File not found. Please run the tuning curve computation script first.")
    mname = 'TX140'
    #random_mname = ['TX144']

    ax = plt.subplot(gs)

    cat = 'light_static'
    idx = 1

    train_data = category_train_data[cat][mname]
    test_data = category_test_data[cat][mname]

    corr = -np.inf
    while not corr >= 0.05:
        # select a random neuron idx
        neuron_idx = np.random.choice(train_data.shape[0])
        # Flatten data to compare
        train_flat = train_data[neuron_idx, :]
        test_flat = test_data[neuron_idx, :]
        corr = np.corrcoef(train_flat, test_flat)[0, 1]
    ax.scatter(train_flat, test_flat, alpha=0.4, s=1.5, edgecolors='none',
                color=rand_stat_colors[idx], label=cat if mname == list(category_train_data[cat].keys())[0] else None)
    # Plot correlation
    ax.text(0.05, 1.8-idx*0.25, f'{cat.split("_")[-1]} r: {corr:.2f}', color='k', 
            fontsize=LABEL_FONTSIZE-3)
    ax.axis("square")

    ax.plot([-1, 2], [-1, 2], 'k--', lw=1)  # identity line
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    # suggest a label for trials split into halves
    ax.set_xlabel('Avg. response\n(train set)')
    ax.set_ylabel('Avg. response\n(test set)')    
    ax.legend(frameon=False, markerscale=4)
    # set xticks
    ax.set_xticks([0, 2], ['0', '2'])
    ax.set_yticks([0, 2], ['0', '2'])
    remove_spines(ax)
    # add left and bottom spines
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    # set pos
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+.005, pos[1]+.03, pos[2], pos[3]])
    ax.set_title('example neuron', fontsize=LABEL_FONTSIZE-2)

def plot_correlation_hist(gs):
    from scipy.stats import ttest_ind

    # Load correlation data if already computed
    if os.path.exists('/media/Atika/tuning_curve_data/correlation_tuning_light_random_static.npy'):
        corr_category = np.load('/media/Atika/tuning_curve_data/correlation_tuning_light_random_static.npy', allow_pickle=True).item()
    else:
        print("File not found. Please run the tuning curve computation script first.")

    # Prepare correlation data
    corr_random = np.hstack(corr_category['light_random']).flatten()
    corr_static = np.hstack(corr_category['light_static']).flatten()

    # Compute means
    mean_random = np.mean(corr_random)
    mean_static = np.mean(corr_static)

    # t-test (independent)
    t_stat, p_val = ttest_ind(corr_random, corr_static)

    # Start plotting
    ax = plt.subplot(gs)
    #hist_kws = {'alpha': 0.5, 'edgecolor': 'none', 'stat': 'probability', 'bins': 50}

    # Plot histograms
    # --- shared bin edges ---
    nbins = 50
    all_corr = np.concatenate([corr_random, corr_static])
    bin_edges = np.histogram_bin_edges(all_corr, bins=nbins)

    hist_kws = {
        'alpha': 0.5,
        'edgecolor': 'none',
        'stat': 'probability',
        'bins': bin_edges
    }

    # Plot histograms with identical bins + normalization
    sns.histplot(
        corr_random,
        label='Random',
        color=rand_stat_colors[0],
        ax=ax,
        **hist_kws
    )

    sns.histplot(
        corr_static,
        label='Static',
        color=rand_stat_colors[1],
        ax=ax,
        **hist_kws
    )

    #sns.histplot(corr_random, label='Random', color=rand_stat_colors[0], **hist_kws, ax=ax)
    #sns.histplot(corr_static, label='Static', color=rand_stat_colors[1], **hist_kws, ax=ax)

    print("Number of neurons (random): ", len(corr_random))
    print("Number of neurons (static): ", len(corr_static))
    # Axis labels
    ax.set_xlabel('Correlation between\nneural tuning curves')
    ax.set_ylabel('Probability')
    ax.set_xlim(-0.02, 0.1)

    # Get y max for scaling lines and bracket
    ymax = ax.get_ylim()[1]

    # Draw mean lines
    ax.vlines(mean_random, ymin=0, ymax=ymax*0.95, colors=rand_stat_colors[0], linestyles='dashed') 
    ax.vlines(mean_static, ymin=0, ymax=ymax*0.95, colors=rand_stat_colors[1], linestyles='dashed')

    # Add mean value text
    ax.text(mean_random + 0.002, ymax*0.95, f"avg. r={mean_random:.3f}", color=rand_stat_colors[0], fontsize=LABEL_FONTSIZE-4)
    ax.text(mean_static + 0.002, ymax*0.85, f"avg. r={mean_static:.3f}", color=rand_stat_colors[1], fontsize=LABEL_FONTSIZE-4)

    """
    # Draw significance bracket
    x1, x2 = mean_random, mean_static
    y, h = ymax*1.04, ymax*0.03
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')

    # Add significance stars
    if p_val < 0.001:
        signif = '***'
    elif p_val < 0.01:
        signif = '**'
    elif p_val < 0.05:
        signif = '*'
    else:
        signif = 'n.s.'
    ax.text((x1+x2)/2, y+h+0.01, signif, ha='center', fontsize=LABEL_FONTSIZE-2)
    """

    # Reduce yticks
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[::2])
    ax.tick_params(axis='both', which='major', labelsize=LABEL_FONTSIZE-2)

    # Custom legend
    legend = ax.legend(frameon=False, fontsize=LABEL_FONTSIZE-2, loc='center right', handlelength=0)
    for j in range(len(legend.get_texts())):
        legend.get_texts()[j].set_color(rand_stat_colors[j])
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor('none')
    for line in legend.get_lines():
        line.set_linewidth(0)

    # Adjust subplot position
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.02, pos[1]+.07, pos[2], pos[3]*.7])


def plot_tuning_curve(gs):

    light_random_dbs = pdh.get_light_random_info()
    light_static_dbs = pdh.get_light_static_info()
    base_folders = [
        f'{BASE_FOLDER}light_recordings_random',
        f'{BASE_FOLDER}light_recordings_static']
    dbs = [light_random_dbs, light_static_dbs]
    category = ['light_random', 'light_static']
    saccade_data = np.load(f'{BASE_FOLDER}saccade_times.npy', allow_pickle=True).item()

    # Load tuning data
    if os.path.exists('/media/Atika/tuning_curve_data/tuning_curve.npy'):
        category_curve = np.load('/media/Atika/tuning_curve_data/tuning_curve.npy', allow_pickle=True).item()
        relative_pupil_position = np.load('/media/Atika/tuning_curve_data/relative_pupil_position.npy')
    else:
        print("File not found. Please run the tuning curve computation script first.")
    # Plot tuning curves
    ax = plt.subplot(gs)
    for idx, cat in enumerate(category[-1::-1]): # plot static first then random
        average = category_curve[cat]
        mean_curve = np.nanmean(average, axis=0)  # Compute mean across trials
        sem_curve = np.nanstd(average, axis=0) / np.sqrt(average.shape[0])  # Compute SEM
        ax.fill_between(range(len(mean_curve)), 
                            mean_curve - sem_curve, 
                            mean_curve + sem_curve, 
                            alpha=0.6, color=rand_stat_colors[category.index(cat)])
        ax.plot(range(len(mean_curve)), mean_curve, label=cat.split('_')[1], color=rand_stat_colors[category.index(cat)])
    ax.set_xlabel('Relative pupil position\n(Avg.°, binned)', fontsize=LABEL_FONTSIZE-1)
    ax.set_ylabel('Avg. response \n(test data; normalized)', labelpad=-2, fontsize=LABEL_FONTSIZE-1)
    
    
    xticks = np.nanmean(relative_pupil_position, axis=0)
    # desired labels in pupil-degree units
    desired_tick_vals = [-10, 0, 10]
    # ignore NaNs and find the index in padded_xticks closest to each desired value
    mask = ~np.isnan(xticks)
    valid_idx = np.where(mask)[0]        # indices of non-NaN entries
    valid_vals = xticks[mask]     # their values

    tick_indices = []
    for v in desired_tick_vals:
        # index in valid_vals closest to v, mapped back to original padded indices
        closest_in_valid = np.argmin(np.abs(valid_vals - v))
        tick_indices.append(int(valid_idx[closest_in_valid]))
    # set ticks at those index positions, but label them with the pupil-degree values
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f"{v}" for v in desired_tick_vals], fontsize=LABEL_FONTSIZE - 3)
    # set xlim
    xlim_vals = [-15, 15]
    xlim_indices = []
    for v in xlim_vals:
        closest_in_valid = np.argmin(np.abs(valid_vals - v))
        xlim_indices.append(int(valid_idx[closest_in_valid]))
    ax.set_xlim(xlim_indices[0], xlim_indices[1])

    # set yticks at 0, 0.5, 1
    ax.set_yticks([0.5, 0.8], ['0.5', '0.8'])
    ax.tick_params(axis='both', which='major', labelsize=LABEL_FONTSIZE-3)

    # Adjust pos
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.02, pos[1]+.07, pos[2], pos[3]*.7])

    legend=ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.25, 1.04))

    for j in range(len(legend.get_texts())):
        legend.get_texts()[j].set_color(rand_stat_colors[rand_stat_color_names.index(legend.get_texts()[j].get_text())])
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)
    legend.get_frame().set_facecolor('none')
    

def plot_varexp_curve(recording_type):

    if recording_type == 'static':
        base_folder = f'{BASE_FOLDER}light_recordings_static'
        nonlinear_results = np.load(f"{base_folder}/nonlinear_lightstatic_results_4500.npy", allow_pickle=True).item()
        dbs = pdh.get_light_static_info()
    else:
        base_folder = f'{BASE_FOLDER}light_recordings_random'
        nonlinear_results = np.load(f"{base_folder}/nonlinear_lightrandom_results_4500.npy", allow_pickle=True).item()
        dbs = pdh.get_light_random_info()

    # remove TX150 from nonlinear_results
    nonlinear_results = {k: v for k, v in nonlinear_results.items() if "TX150" not in k}
    mouse_list = np.unique([entry['mname'] for entry in dbs])
    mouse_list = [mouse for mouse in mouse_list if "TX150" not in mouse] # remove TX150 from mouse_list

    fig = plt.figure(figsize=(3, 5), dpi=150)  # Set the figure size and DPI
    # Define the figure with specified size and higher DPI for better quality
    axs = plt.subplot(111)
    cumsum_varexp = {} # cumulative sum of variance explained for each behavior type

    # Iterate over each mouse in the mouse list
    for mi, mname in enumerate(mouse_list):
        vexp_list = []
        vexp_pc_list = []
        
        # Extract values for each measurement
        input_labels = []
        for behav_input in nonlinear_results[mname].keys():
            if behav_input == 'fullface':
                continue
            vexp = nonlinear_results[mname][behav_input]['vexp_before_carbachol']   # vexp_after_carbachol vexp_before_carbachol
            vexp_pc = nonlinear_results[mname][behav_input]['vexp_pc_before_carbachol'] #  vexp_pc_after_carbachol vexp_pc_before_carbachol
            vpcs = nonlinear_results[mname][behav_input]['vpcs_before_carbachol'] # vpcs_after_carbachol  vpcs_before_carbachol
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
    print(input_labels)
    colors = ['brown', 'green', 'darkorange', 'red']
    labels = ['Pupil area', 'Pupil position', 'Face w/o eye', 'Face w/o eye +\npupil position']
    ordered_labels = ['pupil_area', 'pupil', 'face_eye', 'face_eye_pupil']
    for j in range(mean_vexp_array.shape[1]):
        idx = input_labels.index(ordered_labels[j])
        axs.semilogx(np.arange(1, 129), mean_vexp_array[:,idx]*100, linewidth=3, label=labels[j], color=colors[j])
    for std_idx in range(sem_vexp_array.shape[1]):
        idx = input_labels.index(ordered_labels[std_idx])
        axs.fill_between(np.arange(1, 129), (mean_vexp_array[:,idx] - sem_vexp_array[:,idx])*100, (mean_vexp_array[:,idx] + sem_vexp_array[:,idx])*100, alpha=0.1, color=colors[std_idx])

    axs.set_xlabel("Neural activity principal\ncomponent", fontsize=LABEL_FONTSIZE-1)
    axs.set_ylabel("% Cumulative variance explained\n(test data)")
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_xticklabels([1, 10, 100])
    axs.set_xticks([1, 10, 100])
    axs.set_ylim([0, 35])
    axs.set_yticklabels([0, 10, 20, 30])
    axs.set_yticks([0, 10, 20, 30])

    # change the legends to remove lines and have same color as the plot
    legend=axs.legend(loc='center left', bbox_to_anchor=(-.1,.8), fontsize=LABEL_FONTSIZE-2, frameon=False)
    for j in range(len(legend.get_texts())):
        legend.get_texts()[j].set_color(colors[j])
    legend.get_frame().set_linewidth(0)
    for line in legend.get_lines():
        line.set_linewidth(0)
    legend.get_frame().set_facecolor('none')
    
    # Adjust position
    pos = axs.get_position().bounds
    axs.set_position([pos[0]+.03, pos[1]+.075, pos[2], pos[3]*.85])
    axs.set_title(f'{recording_type.capitalize()} recording; n_kp=250', fontsize=LABEL_FONTSIZE)

    plt.show()


def plot_osi_histogram(gs):
    # Load and prepare data
    threshold = 0.5
    random_osi = []
    static_osi = []
    colors = ['#6a0dad', '#008080']

    base_folder = f'{BASE_FOLDER}light_recordings_random'
    lightrandom_nonlinear_results = np.load(f"{base_folder}/nonlinear_lightrandom_results_4500.npy", allow_pickle=True).item()
    base_folder = f'{BASE_FOLDER}light_recordings_static'
    lightstatic_nonlinear_results = np.load(f"{base_folder}/nonlinear_lightstatic_results_4500.npy", allow_pickle=True).item()
    inputs = [lightrandom_nonlinear_results, lightstatic_nonlinear_results]
    input_names = ['Light random', 'Light static']

    # Collect data (same as before)
    for j in range(len(inputs)):
        data = inputs[j]
        num_animals = len(data)
        for i in range(num_animals):
            mname = list(data.keys())[i]
            if "TX150" in mname:  # Skip TX150 because it has a different sampling rate
                continue
            if "static" in input_names[j]:
                osi_file = glob(f"/home/asyeda/pupil_analysis/light_recordings_static/{mname}_*osi.npy")[0]
                osi_data = np.load(osi_file, allow_pickle=True).item()
                static_osi.extend(osi_data['osi'])
            elif "random" in input_names[j]:
                osi_file = glob(f"/home/asyeda/pupil_analysis/light_recordings_random/{mname}_*osi.npy")[0]
                osi_data = np.load(osi_file, allow_pickle=True).item()
                random_osi.extend(osi_data['osi'])

    random_osi = np.array(random_osi)
    static_osi = np.array(static_osi)

    # Create figure with primary axis
    ax1 = plt.subplot(gs)

    # Manually adjust ax_HK height and vertical position
    orig_pos = ax1.get_position()
    new_height = 0.75 * orig_pos.height  # % of original height
    new_y = orig_pos.y0 + (orig_pos.height - new_height) / 2  # center vertically

    ax1.set_position([orig_pos.x0, new_y, orig_pos.width, new_height])
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.02, pos[1], pos[2]*1.35, pos[3]])

    # Plot histograms (now with probability normalization)
    bin_edges = np.linspace(-0.3, 1.1, 25)
    hist_kws = {'alpha': 0.3, 'edgecolor': 'none', 'stat': 'probability'}  # Changed to 'probability'    
    sns.histplot(random_osi, bins=bin_edges, color=colors[0], 
                label='Random', ax=ax1, kde=False, **hist_kws)
    sns.histplot(static_osi, bins=bin_edges, color=colors[1], 
                label='Static', ax=ax1, kde=False, **hist_kws)

    # Add threshold line
    ax1.axvline(threshold, color='black', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Threshold ({threshold})')

    # Formatting
    ax1.set_xlabel('Orientation Selectivity Index\n(OSI)')
    ax1.set_ylabel('Probability')  # Changed y-axis label
    ax1.set_xlim(-0.3, 1.1)
    ax1.set_ylim(0, None)
    ax1.tick_params(axis='both', which='major')

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=3, label='Random'),
        Line2D([0], [0], color=colors[1], lw=3, label='Static'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, label=f'Threshold={threshold}')
    ]
    ax1.legend(handles=legend_elements, frameon=False, loc='lower right', bbox_to_anchor=(1.2, 0.2), fontsize=LABEL_FONTSIZE-4)

    # Add inset for proportion above threshold
    ax_inset = inset_axes(ax1, width="25%", height="40%", loc='upper right')

    # Calculate proportions
    random_above = np.sum(random_osi >= threshold)
    random_total = len(random_osi)
    static_above = np.sum(static_osi >= threshold)
    static_total = len(static_osi)

    random_prop = random_above/random_total
    static_prop = static_above/static_total

    # Bar plot with exact counts
    conditions = ['Random', 'Static']
    props = [random_prop, static_prop]
    bars = ax_inset.bar(conditions, props, color='blue', alpha=0.6)

    # Format inset
    ax_inset.set_ylabel('Proportion > threshold', fontsize=LABEL_FONTSIZE-4)
    ax_inset.tick_params(axis='both', which='major', labelsize=LABEL_FONTSIZE-4, rotation=90)
    ax_inset.set_ylim(0, max(props)*1.2)
    ax_inset.set_yticks([])

    # Add exact numbers on bars
    for bar, prop, count, total in zip(bars, props, [random_above, static_above], [random_total, static_total]):
        height = bar.get_height()
        ax_inset.text(bar.get_x() + bar.get_width()/2., height,
                    f"{prop:.1%}",
                    ha='center', va='bottom', fontsize=LABEL_FONTSIZE-5)

def zscore_neural_data(data, axis=1):
    data -= data.mean(axis=axis, keepdims=True)
    std = ((data ** 2).mean(axis=axis, keepdims=True) ** 0.5)
    std[std == 0] = 1
    data /= std
    return data