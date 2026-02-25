
from fig3 import get_light_random_info, get_light_static_info
from fig_utils import *
import matplotlib.pyplot as plt 
import os, sys
import numpy as np
from scipy.stats import pearsonr
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib as mpl
import itertools
from scipy.stats import ttest_rel
from matplotlib.lines import Line2D

# Plot signal correlation, face-eye varexp, pupil varexp, tuning correlation

LABEL_FONTSIZE = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = LABEL_FONTSIZE
plt.rcParams['axes.titlesize'] = LABEL_FONTSIZE
plt.rcParams['axes.labelsize'] = LABEL_FONTSIZE
plt.rcParams['xtick.labelsize'] = LABEL_FONTSIZE-1
plt.rcParams['ytick.labelsize'] = LABEL_FONTSIZE-1
plt.rcParams['legend.fontsize'] = LABEL_FONTSIZE
pupil_label_color = 'green'
face_eye_label_color = 'orange'
face_eye_pupil_label_color = 'red'
rand_stat_colors = ['#6a0dad', '#008080']
rand_stat_color_names = ['random', 'static']

# Set folder
BASE_FOLDER = '/home/asyeda/stringerlab_dm11/eye-vs-face/'

def plot():
    fig = plt.figure(figsize=(14, 7), dpi=300)
    gs = GridSpec(2, 1, figure=fig)

    plot_maps_summary(fig, gs[0, 0], 'static')
    plot_maps_summary(fig, gs[1, 0], 'random')

    plt.savefig(f'{BASE_FOLDER}figures/signal_correlation_maps_summary.png', dpi=300, bbox_inches='tight')    
    plt.show()


def plot_num_neurons_brain_region():
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, wspace=0.4)

    # --- Number of neurons per region: static ---
    ax = plt.subplot(gs[0])
    plot_num_neurons_per_region(ax, 'static', 'Number of neurons (static)')

    # --- Number of neurons per region: random ---
    ax = plt.subplot(gs[1])
    plot_num_neurons_per_region(ax, 'random', 'Number of neurons (random)')

    plt.tight_layout()
    plt.show()

def plot_num_neurons_per_region(ax, recording_type, ylabel):
    if recording_type == 'static':  
        dbs = get_light_static_info()
    else:
        dbs = get_light_random_info()

    mouse_list = np.unique([db['mname'] for db in dbs])
    mouse_list = [m for m in mouse_list if m != "TX150"]

    region_names = ['V1', 'medial', 'anterior', 'lateral']
    num_regions = len(region_names)

    # --- grey lines per mouse ---
    all_mice_data = []
    for mname in mouse_list:
        datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]
        if recording_type == 'static':
            blk = '2' if mname != "TX143" else '3'
        else:
            blk = '1'
        signal_variance_map = np.load(
            f"{BASE_FOLDER}data/retinotopy/{mname}_{datexp}_{blk}_signal_variance_map.npy",
            allow_pickle=True
        ).item()
        iarea = signal_variance_map['iarea']

        region_counts = []
        for region in region_names:
            region_mask = get_region_idx(iarea, region)
            val = np.sum(region_mask)
            region_counts.append(val)

        all_mice_data.append(region_counts)
        if mname in ['FXa1', 'FXa8', 'FXa9', 'FXa10']:
            ax.plot(np.arange(num_regions), region_counts, '-o', color='r', alpha=0.8, markersize=3)
        else:
            ax.plot(np.arange(num_regions), region_counts, '-o', color='0.7', alpha=0.8, markersize=3)

    all_mice_data = np.array(all_mice_data)
    mean_vals = np.nanmean(all_mice_data, axis=0)
    sem_vals = np.nanstd(all_mice_data, axis=0, ddof=1) / np.sqrt(len(mouse_list))

    # --- black mean line ---
    ax.plot(np.arange(num_regions), mean_vals, '-o', color='k', linewidth=2.5, markersize=5)

    # --- colored region mean dots ---
    for r_idx, region in enumerate(region_names):
        ax.plot(r_idx, mean_vals[r_idx], 'o', markersize=4, markeredgecolor='k', color='k')
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(np.arange(num_regions))
    ax.set_xticklabels(region_names, ha='right', rotation=30)
    #ax.set_ylim(0, 2500)


def plot_maps_summary(fig, gs, recording_type):
    if recording_type == 'static':
        nonlinear_results = np.load(f"{BASE_FOLDER}data/nonlinear_lightstatic_results_udcnv.npy", allow_pickle=True).item()
        dbs = get_light_static_info() 
        mouse_list = np.unique([db['mname'] for db in dbs])
        # remove TX150 b/c different sampling rate recording
        mouse_list = [m for m in mouse_list if m != "TX150"]
        mname = 'TX143'
        blk = '2' if mname != "TX143" else '3'
    else:
        nonlinear_results = np.load(f"{BASE_FOLDER}data/nonlinear_lightrandom_results_udcnv.npy", allow_pickle=True).item()
        dbs = get_light_random_info()
        mouse_list = np.unique([db['mname'] for db in dbs])
        mname = 'TX143'
        blk='1'

    axes = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs, wspace=0.25)
    datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]
    signal_variance_map = np.load(f"{BASE_FOLDER}data/retinotopy/{mname}_{datexp}_{blk}_signal_variance_map.npy", allow_pickle=True).item()
    xy, signal_variance, out = signal_variance_map['xy'], signal_variance_map['signal_variance'], signal_variance_map['out']    
    behav_data = nonlinear_results[mname]

    # === Define colormaps and normalizations ===
    # Signal variance: symmetric around 0
    cmap_signal = mpl.cm.bwr
    norm_signal = mpl.colors.Normalize(vmin=-0.2, vmax=0.2)

    # Face-eye & pupil variance: always positive, 0 → 0.2
    cmap_var = mpl.cm.viridis  # (you can pick plasma, inferno, etc.)
    norm_var = mpl.colors.Normalize(vmin=0, vmax=5)

    # Signal variance
    ax = plt.subplot(axes[0])
    sc1 = ax.scatter(xy[:,1], xy[:,0], c=signal_variance, s=1,
                    cmap=cmap_signal, norm=norm_signal, edgecolors='none')
    if recording_type == 'static':
        ax.set_title('Signal variance', loc='center')
    for j in range(len(out)):
        ax.plot(out[j][:,1], out[j][:,0], '-k')
    remove_spines(ax)
    ax.set_ylabel(f'{recording_type.capitalize()}', fontsize=LABEL_FONTSIZE, fontdict={'weight':'bold'})

    # Pupil
    ax = plt.subplot(axes[1])
    sc2 = ax.scatter(xy[:,1], xy[:,0], c=behav_data['pupil']['varexp_neurons_before_carbachol']*100,
                    s=1, cmap=cmap_var, norm=norm_var, edgecolors='none')
    if recording_type == 'static':
        ax.set_title('Pupil position', loc='center', fontdict={'color':'{}'.format(pupil_label_color)})
    for j in range(len(out)):
        ax.plot(out[j][:,1], out[j][:,0], '-k')
    remove_spines(ax)

    # Face w/o eye position
    ax = plt.subplot(axes[2])
    sc3 = ax.scatter(xy[:,1], xy[:,0], c=behav_data['face_eye']['varexp_neurons_before_carbachol']*100,
                s=1, cmap=cmap_var, norm=norm_var, edgecolors='none')
    if recording_type == 'static':
        ax.set_title('Face w/o eye', loc='center', fontdict={'color':'{}'.format(face_eye_label_color)})    
    for j in range(len(out)):
        ax.plot(out[j][:,1], out[j][:,0], '-k')
    remove_spines(ax)

    # Face w/o eye + pupil position
    ax = plt.subplot(axes[3])
    sc4 = ax.scatter(xy[:,1], xy[:,0], c=behav_data['face_eye_pupil']['varexp_neurons_before_carbachol']*100,
                    s=1, cmap=cmap_var, norm=norm_var, edgecolors='none')
    if recording_type == 'static':
        ax.set_title('Face w/o eye\n+ pupil position', loc='center', fontdict={'color':'{}'.format(face_eye_pupil_label_color)})
    for j in range(len(out)):
        ax.plot(out[j][:,1], out[j][:,0], '-k')
    remove_spines(ax)

    # === Add colorbars (individually) ===
    x_start = 0.17
    if recording_type == 'random':
        for sc, norm, cmap, label in [
            (sc1, norm_signal, cmap_signal, "Signal variance"),
            (sc2, norm_var, cmap_var, "Variance explained (%)"),
        ]:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar_ax = fig.add_axes([x_start, .52, 0.05, 0.01])  # adjust as needed
            fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=label)
            x_start += 0.3

    # --- summary comparison ---
    ax = plt.subplot(axes[4])
    plot_varexp_comparison(ax, recording_type)
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.04, pos.y0+.04, pos.width, pos.height * 0.8])
    if recording_type == "random":
        ax.set_xlabel('Brain region', fontsize=LABEL_FONTSIZE)

def plot_varexp_comparison(ax, recording_type):
    """Plot paired comparison of Face w/o eye vs Pupil variance explained per region."""
    if recording_type == 'static':  
        nonlinear_results = np.load(f"{BASE_FOLDER}data/nonlinear_lightstatic_results_udcnv.npy", allow_pickle=True).item()
        dbs = get_light_static_info()
    else:
        nonlinear_results = np.load(f"{BASE_FOLDER}data/nonlinear_lightrandom_results_udcnv.npy", allow_pickle=True).item()
        dbs = get_light_random_info()

    mouse_list = [m for m in np.unique([db['mname'] for db in dbs]) if m != "TX150"]
    region_names = ['V1', 'medial', 'anterior', 'lateral']

    face_data, pupil_data, all_data = [], [], []

    for mname in mouse_list:
        datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]
        if recording_type == "static":
            blk = '2' if mname != "TX143" else '3'
        else:
            blk = '1'
        signal_variance_map = np.load(
            f"{BASE_FOLDER}data/retinotopy/{mname}_{datexp}_{blk}_signal_variance_map.npy",
            allow_pickle=True
        ).item()
        iarea = signal_variance_map['iarea']
        behav_data = nonlinear_results[mname]

        face_vals, pupil_vals, all_vals = [], [], []
        for region in region_names:
            region_mask = get_region_idx(iarea, region)
            face_vals.append(np.nanmean(behav_data['face_eye']['varexp_neurons_before_carbachol'][region_mask]))
            pupil_vals.append(np.nanmean(behav_data['pupil']['varexp_neurons_before_carbachol'][region_mask]))
            all_vals.append(np.nanmean(behav_data['face_eye_pupil']['varexp_neurons_before_carbachol'][region_mask]))
        face_data.append(face_vals)
        pupil_data.append(pupil_vals)
        all_data.append(all_vals)

    face_data, pupil_data, all_data = np.array(face_data), np.array(pupil_data), np.array(all_data)
    # convert to percentage
    face_data *= 100
    pupil_data *= 100
    all_data *= 100

    # --- Plot paired points for each region ---
    x_positions = np.arange(len(region_names))
    offset = 0.2  # horizontal spacing between the two conditions

    for i in range(len(mouse_list)):
        ax.plot(
            x_positions, face_data[i], 'o', color=face_eye_label_color, alpha=0.75, markersize=3
        )
        ax.plot(
            x_positions - offset, pupil_data[i], 'o', color=pupil_label_color, alpha=0.75, markersize=3
        )
        ax.plot(
            x_positions + offset, all_data[i], 'o', color=face_eye_pupil_label_color, alpha=0.75, markersize=3
        )
        # connect lines
        for j, x in enumerate(x_positions):
            ax.plot([x - offset, x, x + offset], [pupil_data[i, j], face_data[i, j], all_data[i, j]], color='0.8', alpha=0.5, linewidth=0.8)
            #ax.plot([x - offset, x + offset], [face_data[i, j], pupil_data[i, j]], color='0.8', alpha=0.5, linewidth=0.8)

    if recording_type == "static":
        # Coordinates in axis space (0–1 range)
        x0, y0 = 1.01, 0.9  # anchor point — adjust as needed
        dy = 0.07           # vertical spacing between lines
        ax.text(x0, y0, 'Pupil', 
                transform=ax.transAxes, fontsize=10, color=pupil_label_color, ha='right')
        ax.text(x0, y0-dy, 'Face w/o eye', 
                transform=ax.transAxes, fontsize=10, color=face_eye_label_color, ha='right')
        ax.text(x0, y0 - 2*dy, 'Face w/o eye + Pupil position', 
                transform=ax.transAxes, fontsize=10, color=face_eye_pupil_label_color, ha='right')

    # --- Significance bars ---
    y_max = np.nanmax(np.vstack([face_data, pupil_data, all_data])) #np.vstack([face_mean + face_sem, pupil_mean + pupil_sem]))
    axin = ax.inset_axes([0, 1, 1, 0.2])
    for r_idx, region in enumerate(region_names):
        bar_height = y_max + 0.003
        tval, pval = ttest_rel(face_data[:, r_idx], pupil_data[:, r_idx])
        add_sig_bar(axin, r_idx, r_idx-offset, bar_height, pval)
        bar_height += 0.005
        tval, pval = ttest_rel(face_data[:, r_idx], all_data[:, r_idx])
        add_sig_bar(axin, r_idx, r_idx + offset, bar_height, pval)
        bar_height += 0.005
        tval, pval = ttest_rel(pupil_data[:, r_idx], all_data[:, r_idx])
        add_sig_bar(axin, r_idx-offset, r_idx + offset, bar_height, pval)
        bar_height += 0.005
    remove_spines(axin)

    # --- Styling ---
    ax.set_xticks(x_positions)
    ax.set_xticklabels(region_names, rotation=30, ha='right')
    ax.set_ylabel('Avg. variance explained (%)', fontsize=LABEL_FONTSIZE)
    ax.set_xlim(-0.5, len(region_names) - 0.5)
    if recording_type == "random":
        ax.set_yticks([0, 5], [0, 5])
    else:
        ax.set_yticks([0, 5, 10], [0, 5, 10])

def get_region_idx(iarea, region):
    if region == 'all':
        ix = np.isin(iarea, np.arange(10))
    if region == 'V1':
        ix = np.isin(iarea, [8])
    if region == 'medial':
        ix = np.isin(iarea, [0, 1, 2, 9])
    if region == 'anterior':
        ix = np.isin(iarea, [3, 4])
    if region == 'lateral':
        ix = np.isin(iarea, [5, 6, 7])
    return ix


def add_sig_bar(ax, x1, x2, y, pval, h=0.001):
    """Draw a significance bar between two x-positions with star annotation."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c='k')
    if pval < 0.001:
        stars = '***'
    elif pval < 0.01:
        stars = '**'
    elif pval < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    if stars == 'n.s.':
        ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom', fontsize=9)
    else:
        ax.text((x1 + x2) / 2, y - h*2, stars, ha='center', va='bottom', fontsize=12)


def plot_varexp(ax, recording_type, behav_key, ylabel):
    if recording_type == 'static':  
        nonlinear_results_lightstatic = np.load(f"{BASE_FOLDER}data/nonlinear_lightstatic_results_udcnv.npy", allow_pickle=True).item()
        dbs = get_light_static_info()
    else:
        nonlinear_results_lightstatic = np.load(f"{BASE_FOLDER}data/nonlinear_lightrandom_results_udcnv.npy", allow_pickle=True).item()
        dbs = get_light_random_info()

    mouse_list = np.unique([db['mname'] for db in dbs])
    mouse_list = [m for m in mouse_list if m != "TX150"]

    region_names = ['V1', 'medial', 'anterior', 'lateral']
    num_regions = len(region_names)

    # --- grey lines per mouse ---
    all_mice_data = []
    for mname in mouse_list:
        datexp = [entry['datexp'] for entry in dbs if entry['mname'] == mname][0]
        if recording_type == 'static':
            blk = '2' if mname != "TX143" else '3'
        else:
            blk = '1'
        signal_variance_map = np.load(
            f"{BASE_FOLDER}data/retinotopy/{mname}_{datexp}_{blk}_signal_variance_map.npy",
            allow_pickle=True
        ).item()
        iarea = signal_variance_map['iarea']
        behav_data = nonlinear_results_lightstatic[mname]

        region_means = []
        for region in region_names:
            region_mask = get_region_idx(iarea, region)
            val = np.nanmean(behav_data[behav_key]['varexp_neurons_before_carbachol'][region_mask])
            region_means.append(val)

        all_mice_data.append(region_means)
        ax.plot(np.arange(num_regions), region_means, '-o', color='0.7', alpha=0.8, markersize=3)

    all_mice_data = np.array(all_mice_data)
    mean_vals = np.nanmean(all_mice_data, axis=0)
    sem_vals = np.nanstd(all_mice_data, axis=0, ddof=1) / np.sqrt(len(mouse_list))

    # --- black mean line ---
    ax.plot(np.arange(num_regions), mean_vals, '-o', color='k', linewidth=2.5, markersize=5)

    # --- colored region mean dots ---
    for r_idx, region in enumerate(region_names):
        ax.plot(r_idx, mean_vals[r_idx], 'o', markersize=4, markeredgecolor='k', color='k')

    ax_inset = ax.inset_axes([0, 1, 1, 0.3])
    # --- paired t-tests between regions ---
    y_max = np.nanmax(mean_vals + sem_vals)
    bar_height = y_max + 0.0035
    step = 0.0035
    for (i, j) in itertools.combinations(range(num_regions), 2):
        tval, pval = ttest_rel(all_mice_data[:, i], all_mice_data[:, j])
        #print(pval, region_names[i], region_names[j])
        add_sig_bar(ax_inset, i, j, bar_height, pval)
        bar_height += step
    remove_spines(ax_inset)

    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(np.arange(num_regions))
    ax.set_xticklabels(region_names, ha='right', rotation=30)
    ax.set_ylim(0, 0.076)
    ax.set_yticks(np.arange(0, ax.get_ylim()[1], 0.035))
    ax.legend(frameon=False, fontsize=8)

def plot_brain_region_summary():
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, wspace=0.4)

    # --- Pupil variance explained ---
    ax = plt.subplot(gs[0])
    plot_varexp(ax, 'pupil', 'Avg. pupil variance explained')

    # --- Face-eye variance explained ---
    ax = plt.subplot(gs[1])
    plot_varexp(ax, 'face_eye', 'Avg. Face w/o eye variance explained')

    plt.tight_layout()
    plt.show()

def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
