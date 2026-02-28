# This script gets PSTH of keypoints for all mice
# It performs the following steps:
# 1. Load data from all mice
# 2. Get saccades
# 3. Get PSTH 
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pupil_dark_helpers as pdh
import numpy as np
import nonlinear_modelv3
import helper_functions
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import matplotlib.pyplot as plt
from scipy import interpolate  # For resampling


BASE_FOLDER = os.getcwd().replace('analysis/code', '') # base folder for all data
print("Base folder: ", BASE_FOLDER) # this should be parent folder called eye-vs-faces

light_random_dbs = pdh.get_light_random_info()
light_static_dbs = pdh.get_light_static_info()
dbs = [light_random_dbs, light_static_dbs]
category = ['light_random', 'light_static']
saccade_data = np.load(f'{BASE_FOLDER}/data/saccade_data/saccade_times.npy', allow_pickle=True).item()

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

mean_running_speed_all = []
mean_running_speed_random_all = []
# Define target sampling rate (Hz) for all datasets
TARGET_SR = 50  # Choose a common sampling rate (e.g., 50 Hz)
padding_seconds = 1  # seconds before/after saccade

for db_idx, db in enumerate(dbs):
    # Get mouse list
    mouse_list = np.unique([entry['mname'] for entry in db])
    mouse_list = mouse_list[mouse_list != 'TX150']
    for mname in mouse_list:
        print('Processing mouse: ', mname)
        if (category[db_idx] == 'light_random') and (mname not in ['FXa10', 'FXa8', 'FXa9', 'FXa1']):
            print('Skipping mouse ', mname, ' for light random category')
            continue
        # Get datexp 
        datexp = [entry['datexp'] for entry in db if entry['mname'] == mname][0]
        # Load data
        _, _, _, _, _, _, _, npz_data = nonlinear_modelv3.load_data(f'{BASE_FOLDER}/data/npz_data/', mname, datexp) # different base folder for these mice
       
        # segment the data into blocks
        block_frames = npz_data['cam_frames_list']
        
        # Get number of blocks in the recording
        blocks = list(npz_data['tcam'].item())
        num_blocks = len(blocks)

        for blk_idx in range(1):
            blk = blocks[blk_idx].split('blk')[1]
            print('Processing block: ', blk)
            # Get pupil timestamp
            pupil_timestamp = npz_data['tcam'].item()[list(npz_data['tcam'].item())[blk_idx]]
            pupil_x = helper_functions.segment_data(npz_data["pupil_x_small"], block_frames, blk_idx)
            pupil_y = helper_functions.segment_data(npz_data["pupil_y_small"] , block_frames, blk_idx)
            blink = helper_functions.segment_data(npz_data["blink_small"], block_frames, blk_idx)
            pupil_xy = np.vstack((pupil_x, pupil_y)).T

            # Get running data
            running_speed = helper_functions.segment_data(npz_data["running_speed"], block_frames, blk_idx)
            running_speed = np.abs(running_speed)
            running_timestamp = helper_functions.segment_data(npz_data["running_time"], block_frames, blk_idx)
            blink_baselined_threshold = float([entry['blink_baselined_threshold'] for entry in db if entry['mname'] == mname and entry['blk']==blk][0])

            # Baseline correction for blink artifacts
            blink_baselined = gaussian_filter(blink, 10)
            blink_baselined = minimum_filter1d(blink_baselined, 5)
            blink_baselined = maximum_filter1d(blink_baselined, 5)
            blink_baselined = blink - blink_baselined
            pupil_x_filtered = helper_functions.interpolate_pupil_blink(pupil_x, blink_baselined, blink_baselined_threshold)
            pupil_y_filtered = helper_functions.interpolate_pupil_blink(pupil_y, blink_baselined, blink_baselined_threshold)

            # Filter out high frequency noise
            tmin = np.min(pupil_timestamp)
            cam_time = (pupil_timestamp - tmin) * 24 * 60 * 60
            sampling_rate = (1 / np.mean(np.diff(cam_time))).astype(int)

            exact_saccades = saccade_data[category[db_idx]][mname][blk] # Get exact saccade indices from precomputed data
            
            # Truncate running data where timestamp diff is negative due to saving issue leading to series of zero timestamps
            neg_diffs = np.where(np.diff(running_timestamp) < 0)[0]
            if len(neg_diffs) > 0:
                first_neg_idx = neg_diffs[0] + 1
                running_timestamp = running_timestamp[:first_neg_idx]
                running_speed = running_speed[:first_neg_idx]

            # Smooth running speed to reduce noise and if 5 consecutive samples are zero, set to zero
            running_speed = gaussian_filter(running_speed, sigma=2)
            zero_runs = np.where(running_speed == 0)[0]
            for zr in zero_runs:
                if np.all(np.isin(np.arange(zr, zr+5), zero_runs)):
                    running_speed[zr:zr+5] = 0

            # ~~~~~~~~ Plot mean PSTH of whisker, nose and eye keypoints before and after saccades ~~~~~~~~~
            tmin = np.min(running_timestamp)
            run_time = (running_timestamp - tmin) * 24 * 60 * 60
            running_sampling_rate = (1 / np.mean(np.diff(run_time))).astype(int)
            print(">>>>Running Sampling Rate: ", running_sampling_rate)
            original_padding = int(padding_seconds * running_sampling_rate)  # padding in original samples
            
            # Calculate padding in target samples
            target_padding = int(padding_seconds * TARGET_SR)

            # List to store values
            psth_running_all = []
            psth_running_random = []

            # Create array to mark saccade times in running data to avoid overlap with random times
            saccade_times_running = np.zeros(len(running_timestamp), dtype=bool)
            for sac in exact_saccades:
                max_vel_pupil_idx = sac[-1]
                max_vel_pupil_time = pupil_timestamp[max_vel_pupil_idx]
                start = (np.abs(running_timestamp - max_vel_pupil_time)).argmin()
                saccade_times_running[start-original_padding:start+original_padding] = True

            # Iterate through each saccade
            for sac in exact_saccades:
                max_vel_pupil_idx = sac[-1] # max velocity index for pupil data sampled at different rate; get closest index in running data
                max_vel_pupil_time = pupil_timestamp[max_vel_pupil_idx]
                # Find closest index in running data
                start = (np.abs(running_timestamp - max_vel_pupil_time)).argmin()
                #start, end = sac[0], sac[1]
                # Calculate padded start and end
                start_pad = max(0, start - original_padding)
                end_pad = min(len(running_speed)-1, start + original_padding)
                
                # Get the time window in seconds
                time_window = start_pad / running_sampling_rate, end_pad / running_sampling_rate
                
                # Create original time vector for this segment
                num_original_time_points = end_pad - start_pad
                original_time = np.linspace(time_window[0], time_window[-1], num_original_time_points)
                
                # Create target time vector (resampled)
                target_time = np.linspace(time_window[0], time_window[-1], target_padding*2)
                    
                # Compute speed (distance/time) for whisker, nose, and eye centroids
                psth_running = running_speed[start_pad:end_pad]

                # Resample each PSTH to target sampling rate
                if len(psth_running) > 1:
                    f_run = interpolate.interp1d(original_time, psth_running, kind='linear', fill_value='interpolate')
                    psth_running = f_run(target_time)
                psth_running_all.append(psth_running)

                # Pick a random time that doesn't overlap with saccades and start_pad:end_pad
                random_start = np.random.randint(0, len(running_speed)-(original_padding * 2))
                random_end = random_start + (original_padding * 2)
                
                # Pick a random time that doesn't overlap with saccades and start_pad:end_pad
                valid = False
                while not valid:
                    random_start = np.random.randint(0, len(running_speed) - (original_padding * 2))
                    random_end = random_start + (original_padding * 2)

                    # Check overlap with saccade window
                    overlap_with_window = not (random_end <= start_pad or random_start >= end_pad)

                    # Check bounds
                    within_bounds = random_end <= len(running_speed)

                    # Check no overlap with saccade times
                    overlap_with_saccades = np.any(saccade_times_running[random_start:random_end])

                    # Accept if no overlap and correct length
                    if not overlap_with_window and within_bounds and not overlap_with_saccades:
                        valid = True
                
                psth_run_rand = running_speed[random_start:random_end]
                if len(psth_run_rand) > 1:
                    time_window_rand = (random_start / running_sampling_rate, random_end / running_sampling_rate)
                    num_original_time_points = len(psth_run_rand)
                    original_time_rand = np.linspace(time_window_rand[0], time_window_rand[-1], num_original_time_points)#[:psth_run_rand.shape[0]]
                    target_time_rand = np.linspace(time_window_rand[0], time_window_rand[-1], target_padding*2)
                    
                    # Resample random speed to target sampling rate
                    f_run_rand = interpolate.interp1d(original_time_rand, psth_run_rand, kind='linear', fill_value='interpolate')
                    psth_run_rand = f_run_rand(target_time_rand)
                psth_running_random.append(psth_run_rand)
        
            # Convert lists to numpy arrays
            psth_running_random = np.array(psth_running_random)
            psth_running_all = np.array(psth_running_all)

            # Compute mean speed values
            mean_psth_running = np.mean(psth_running_all, axis=0)
            mean_psth_running_random = np.mean(psth_running_random, axis=0)

            # Append to list
            mean_running_speed_all.append(mean_psth_running)
            mean_running_speed_random_all.append(mean_psth_running_random) 

# Now all PSTHs are resampled to TARGET_SR, so we can use target_padding for analysis
padding = target_padding  # This is now consistent across all datasets

print("mean_running_speed_all", np.array(mean_running_speed_all).shape)
# Make a plot saving the post and pre avg. plot as line plot showing each mouse
pre_idx = np.arange(padding-10, padding) # pre-saccade 
post_idx = np.arange(padding, padding+10) # post-saccade
post_avg_running = np.mean(np.array(mean_running_speed_all)[:, post_idx], axis=1)
pre_avg_running = np.mean(np.array(mean_running_speed_all)[:, pre_idx], axis=1)
random_avg_running = np.nanmean(np.array(mean_running_speed_random_all)[:, post_idx], axis=1)

# Plot individual mouse lines
fig, ax = plt.subplots(1, 1, figsize=(4, 7))  
ax.plot([random_avg_running, pre_avg_running, post_avg_running], color='darkorange', alpha=0.25, lw=0.5)
# plot avg
ax.plot([np.mean(random_avg_running), np.mean(pre_avg_running), np.mean(post_avg_running)], 
        marker='o', color='blue', label='Running', lw=2)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Random', 'Pre-saccade', 'Post-saccade'])
ax.set_ylabel('Avg. speed (mm/s)')
ax.legend(loc='upper left', frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Function to add significance bars
def add_significance_bar(ax, x1, x2, y, p, height_factor=0.1):
    """Add significance bar and annotation"""
    ax.plot([x1, x1, x2, x2], 
            [y, y + height_factor, y + height_factor, y], 
            color='black', lw=1)
    p_text = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else f'p={p:.2f}'
    ax.text((x1+x2)/2, y + height_factor*1.1, p_text, 
            ha='center', va='bottom', fontsize=8)

# Perform all pairwise comparisons for whiskers
conditions = ['Random', 'Pre', 'Post']
running_data = [random_avg_running, pre_avg_running, post_avg_running]
print("Statistical Test Results:")
from itertools import combinations
from scipy.stats import ttest_rel

print("\Running Comparisons:")
for (i, j) in combinations([0, 1, 2], 2):
    data1, data2 = running_data[i], running_data[j]
    stat, p = ttest_rel(data1, data2) if len(data1) == len(data2) else (np.nan, np.nan)
    test_type = "Paired t-test"
    
    # Plot significance
    y_max = max(np.mean(data1), np.mean(data2))
    height_offset = 0.01 + (i+j)*0.055  # Stagger heights
    if not np.isnan(p) and p<0.05:
        add_significance_bar(ax, i, j, y_max + height_offset*y_max, p, height_factor=0.01*y_max)
    
    print(f"{conditions[i]} vs {conditions[j]}: {test_type}, stat={stat:.3f}, p={p:.4f}")


# Convert lists to numpy arrays and get mean values
mean_psth_running_speed_all = np.mean(np.array(mean_running_speed_all), axis=0)
mean_psth_running_speed_random_all = np.mean(np.array(mean_running_speed_random_all), axis=0)
sem_psth_running_speed_all = np.std(np.array(mean_running_speed_all), axis=0) / np.sqrt(len(mean_running_speed_all))
sem_psth_running_speed_random_all = np.std(np.array(mean_running_speed_random_all), axis=0) / np.sqrt(len(mean_running_speed_random_all))


# Plot results
fig, ax = plt.subplots(1, 2, figsize=(7, 5), dpi=150, sharey=True)

# Plot mean values and SEM
ax[0].plot(mean_psth_running_speed_all, label="Running", lw=1, color='blue')
ax[0].fill_between(range(len(mean_psth_running_speed_all)), mean_psth_running_speed_all - sem_psth_running_speed_all, mean_psth_running_speed_all + sem_psth_running_speed_all, color='blue', alpha=0.2)
ax[0].legend(loc='upper left', frameon=False)

ax[1].plot(mean_psth_running_speed_random_all, label="Running (Random)", lw=1, color='blue')
ax[1].fill_between(range(len(mean_psth_running_speed_random_all)), mean_psth_running_speed_random_all - sem_psth_running_speed_random_all, mean_psth_running_speed_random_all + sem_psth_running_speed_random_all, color='blue', alpha=0.2)

padding = target_padding  # This is now consistent across all datasets

# Create time vector for x-axis (now using TARGET_SR)
n_samples = len(mean_psth_running_speed_all)
time_vector = (np.arange(n_samples) - padding) / TARGET_SR  # in seconds

# Choose ticks based on time
tick_spacing_sec = 1  # ticks every x seconds
min_time = time_vector[0]
max_time = time_vector[-1]

xtick_labels = np.arange(np.ceil(min_time / tick_spacing_sec) * tick_spacing_sec,
                         np.floor(max_time / tick_spacing_sec) * tick_spacing_sec + tick_spacing_sec,
                         tick_spacing_sec)

# Map these labels back to data indices
xticks = (xtick_labels * TARGET_SR + padding).astype(int)
xtick_labels = np.round(xtick_labels, 1)

# Convert xtick_labels to seconds
#xtick_labels = xtick_labels / sampling_rate
#xtick_labels = np.round(xtick_labels, 1)
ax[0].set_xticks(xticks, xtick_labels)
max_val = ax[0].get_ylim()[1]
min_val = ax[0].get_ylim()[0]
ax[0].set_ylim(min_val-0.1, max_val+0.1)
ax[0].set_xlim(0, len(mean_psth_running_speed_all))
ax[0].vlines(padding, ymin=min_val-.1, ymax=max_val+0.2, color='red', linestyle='--', lw=.8)

ax[1].vlines(padding, ymin=min_val-.1, ymax=max_val+0.2, color='red', linestyle='--', lw=.8)
ax[1].set_xticks(xticks, xtick_labels)
# Add x-axis ticks and labels
ax[0].set_xlabel('Time from saccade (s)')
ax[0].set_ylabel('Speed (mm/s)')
ax[1].set_xlabel('Random window (s)')

# Remove spines
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

fig.suptitle("Light static recordings - PSTH of running speed")
plt.tight_layout()
plt.show()

# Save data for plotting
np.savez(f'{BASE_FOLDER}/data/figure_data/saccade_data/psth_running.npz', mean_psth_running_speed_all=mean_psth_running_speed_all, mean_psth_running_speed_random_all=mean_psth_running_speed_random_all,
         sem_psth_running_speed_all=sem_psth_running_speed_all, sem_psth_running_speed_random_all=sem_psth_running_speed_random_all,
         padding=padding, sampling_rate=TARGET_SR, random_avg_running=random_avg_running, pre_avg_running=pre_avg_running, post_avg_running=post_avg_running)