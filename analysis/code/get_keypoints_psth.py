# This script gets PSTH of keypoints for all mice
# It performs the following steps:
# 1. Load data from all mice
# 2. Get saccades
# 3. Get PSTH of keypoints
import sys, os
import pupil_dark_helpers as pdh
import numpy as np
import nonlinear_modelv3
import helper_functions
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from itertools import combinations
from scipy import interpolate  # For resampling

BASE_FOLDER = os.getcwd().replace('analysis/code', '') # base folder for all data
print("Base folder: ", BASE_FOLDER) # this should be parent folder called eye-vs-faces

dark_dbs = pdh.carbachol_in_dark() 
light_random_dbs = pdh.get_light_random_info()
light_static_dbs = pdh.get_light_static_info()
dbs = [dark_dbs, light_random_dbs, light_static_dbs]
category = ['dark', 'light_random', 'light_static']
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

# Define target sampling rate (Hz) for all datasets
TARGET_SR = 50  # Choose a common sampling rate (e.g., 50 Hz)
padding_seconds = 1 # seconds before/after saccade

mean_psth_whisker_all = []
mean_psth_nose_all = []
mean_psth_eye_all = []
mean_psth_pupilarea_all = []
mean_psth_whisker_random_all = []
mean_psth_nose_random_all = []
mean_psth_eye_random_all = []
mean_psth_pupilarea_random_all = []
sample_rates = []

for db_idx, db in enumerate(dbs):
    # Get mouse list
    mouse_list = np.unique([entry['mname'] for entry in db])
    mouse_list = mouse_list[mouse_list != 'TX150']
    for mname in mouse_list:
        print('>>>> mouse: ', mname, category[db_idx])
        # Get datexp 
        datexp = [entry['datexp'] for entry in db if entry['mname'] == mname][0]
        # Load data
        if mname in ['FXa10', 'FXa8', 'FXa9', 'FXa1']:  # gcamp8 recordings
            _, _, _, _, _, _, _, npz_data = nonlinear_modelv3.load_data(f'{BASE_FOLDER}/data/mouse_data/{mname}/{datexp}/{1}', mname, datexp) # different base folder for these mice
        else:
            _, _, _, _, _, _, _, npz_data = nonlinear_modelv3.load_data(f'{BASE_FOLDER}/data/mouse_data/{mname}/{datexp}', mname, datexp)

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
            print('block: ', blk)
            # Get pupil timestamp
            pupil_timestamp = npz_data['tcam'].item()[list(npz_data['tcam'].item())[blk_idx]]
            pupil_x = helper_functions.segment_data(npz_data["pupil_x_small"], block_frames, blk_idx)
            pupil_y = helper_functions.segment_data(npz_data["pupil_y_small"] , block_frames, blk_idx)
            pupil_area = helper_functions.segment_data(npz_data["pupil_area_small"], block_frames, blk_idx)
            print("block frames", block_frames, blk_idx, pupil_area.shape)
            print("npz_data['pose_x_coord']", npz_data['pose_x_coord'].shape)
            pose_x_coord = helper_functions.segment_data(npz_data['pose_x_coord'].T, block_frames, blk_idx)
            pose_y_coord = helper_functions.segment_data(npz_data['pose_y_coord'].T, block_frames, blk_idx)
            blink = helper_functions.segment_data(npz_data["blink_small"], block_frames, blk_idx)
            pupil_xy = np.vstack((pupil_x, pupil_y)).T

            # replace nan with interpolated values
            nans, x= np.isnan(pupil_area), lambda z: z.nonzero()[0]
            pupil_area[nans]= np.interp(x(nans), x(~nans), pupil_area[~nans])
            
            # zscore pupil area
            pupil_area = (pupil_area - np.nanmean(pupil_area)) / np.nanstd(pupil_area)

            # Filter out high frequency noise
            cutoff_freq = 1.5  # Adjust this value depending on the movement frequency you want to filter out (in Hz)
            tmin = np.min(pupil_timestamp)
            cam_time = (pupil_timestamp - tmin) * 24 * 60 * 60
            sampling_rate = (1 / np.mean(np.diff(cam_time))).astype(int)
            sample_rates.append(sampling_rate)
            print("sampling rate: ", sampling_rate)
            
            exact_saccades = saccade_data[category[db_idx]][mname][blk] # Get exact saccade indices from precomputed data

            # ~~~~~~~~ Plot mean PSTH of whisker, nose and eye keypoints before and after saccades ~~~~~~~~~
            original_padding = int(padding_seconds * sampling_rate)  # padding in original samples
            
            # Calculate padding in target samples
            target_padding = int(padding_seconds * TARGET_SR)
            
            # Extract pose values before and after the saccade
            pose_whisker_x = pose_x_coord[:, -3:]  # Last 3 keypoints for whiskers
            pose_whisker_y = pose_y_coord[:, -3:]  # Last 3 keypoints for whiskers
            pose_nose_x = pose_x_coord[:, 6:10]  # Nose keypoints excluding nosebridge
            pose_nose_y = pose_y_coord[:, 6:10]  # Nose keypoints excluding nosebridge
            pose_eye_x = pose_x_coord[:, 0:4]  # Eye keypoints
            pose_eye_y = pose_y_coord[:, 0:4]  # Eye keypoints

            # Get centroid of whisker, nose, and eye keypoints
            whisker_x_centroid = np.mean(pose_whisker_x, axis=1)
            whisker_y_centroid = np.mean(pose_whisker_y, axis=1)
            whisker_centroid_dist = np.sqrt(np.diff(whisker_x_centroid)**2 + np.diff(whisker_y_centroid)**2) # np.diff(whisker_y_centroid) #

            nose_x_centroid = np.mean(pose_nose_x, axis=1)
            nose_y_centroid = np.mean(pose_nose_y, axis=1)
            nose_centroid_dist = np.sqrt(np.diff(nose_x_centroid)**2 + np.diff(nose_y_centroid)**2) #np.diff(nose_y_centroid) #

            eye_x_centroid = np.mean(pose_eye_x, axis=1)
            eye_y_centroid = np.mean(pose_eye_y, axis=1)
            eye_centroid_dist = np.sqrt(np.diff(eye_x_centroid)**2 + np.diff(eye_y_centroid)**2) # np.diff(eye_y_centroid) #

            # List to store values
            psth_whisker_all = []
            psth_nose_all = []
            psth_eye_all = []
            psth_pupilarea_all = []
            psth_whisker_random = []
            psth_nose_random = []
            psth_eye_random = []
            psth_pupilarea_random = []

            # Create array to mark saccade times to avoid overlap with random times
            saccade_times_pupil = np.zeros(len(pupil_x), dtype=bool)
            for sac in exact_saccades:
                max_vel_pupil_idx = sac[-1]
                saccade_times_pupil[max_vel_pupil_idx-original_padding:max_vel_pupil_idx+original_padding] = True

            # Iterate through each saccade
            for sac in exact_saccades:
                #start, end = sac[0], sac[1]
                start = sac[-1] # max velocity index
                # Calculate padded start and end in original samples
                start_pad = max(0, start - original_padding)
                end_pad = min(len(pupil_x)-1, start + original_padding)
                
                # Get the time window in seconds
                time_window = (start_pad / sampling_rate, end_pad / sampling_rate)
                
                # Compute speed (distance/time) for whisker, nose, and eye centroids
                psth_whisker = whisker_centroid_dist[start_pad:end_pad]
                psth_nose = nose_centroid_dist[start_pad:end_pad]
                psth_eye = eye_centroid_dist[start_pad:end_pad]
                psth_pupilarea = pupil_area[start_pad:end_pad]
                                
                # Create original time vector for this segment
                original_time = np.linspace(time_window[0], time_window[-1], len(psth_whisker))
                
                # Create target time vector (resampled)
                target_time = np.linspace(time_window[0], time_window[-1], target_padding*2)
                
                # Resample each PSTH to target sampling rate
                if len(psth_whisker) > 1:  # Need at least 2 points to interpolate
                    f_whisker = interpolate.interp1d(original_time, psth_whisker, kind='linear', fill_value='interpolate')
                    psth_whisker = f_whisker(target_time)
                    
                    f_nose = interpolate.interp1d(original_time, psth_nose, kind='linear', fill_value='interpolate')
                    psth_nose = f_nose(target_time)
                    
                    f_eye = interpolate.interp1d(original_time, psth_eye, kind='linear', fill_value='interpolate')
                    psth_eye = f_eye(target_time)

                    f_pupilarea = interpolate.interp1d(original_time, psth_pupilarea, kind='linear', fill_value='interpolate')
                    psth_pupilarea = f_pupilarea(target_time)
                
                # Append to lists
                psth_whisker_all.append(psth_whisker)
                psth_nose_all.append(psth_nose)
                psth_eye_all.append(psth_eye)
                psth_pupilarea_all.append(psth_pupilarea)

                # Pick a random time that doesn't overlap with saccades and start_pad:end_pad
                random_start = np.random.randint(0, len(pupil_area)-(original_padding * 2))
                random_end = random_start + (original_padding * 2)
                
                # Pick a random time that doesn't overlap with saccades and start_pad:end_pad
                valid = False
                while not valid:
                    random_start = np.random.randint(0, len(pupil_area) - (original_padding * 2))
                    random_end = random_start + (original_padding * 2)

                    # Check overlap with saccade window
                    overlap_with_window = not (random_end <= start_pad or random_start >= end_pad)

                    # Check bounds
                    within_bounds = random_end <= len(pupil_area)

                    # Check if the random window overlaps with any saccade times
                    overlap_with_saccades = np.any(saccade_times_pupil[random_start:random_end])

                    # Accept if no overlap and correct length
                    if not overlap_with_window and within_bounds and not overlap_with_saccades:
                        valid = True
             
                # Get random segment and resample
                psth_whisker_rand = whisker_centroid_dist[random_start:random_end]
                psth_nose_rand = nose_centroid_dist[random_start:random_end]
                psth_eye_rand = eye_centroid_dist[random_start:random_end]
                psth_pupilarea_rand = pupil_area[random_start:random_end]
                    
                if len(psth_whisker_rand) > 1:
                    time_window_rand = (random_start / sampling_rate, random_end / sampling_rate)
                    num_original_timepoints = len(psth_whisker_rand)
                    original_time_rand = np.linspace(time_window_rand[0], time_window_rand[-1], num_original_timepoints)
                    target_time_rand = np.linspace(time_window_rand[0], time_window_rand[-1], target_padding*2)
                    
                    f_whisker_rand = interpolate.interp1d(original_time_rand, psth_whisker_rand, kind='linear', fill_value='interpolate')
                    psth_whisker_rand = f_whisker_rand(target_time_rand)
                    
                    f_nose_rand = interpolate.interp1d(original_time_rand, psth_nose_rand, kind='linear', fill_value='interpolate')
                    psth_nose_rand = f_nose_rand(target_time_rand)
                    
                    f_eye_rand = interpolate.interp1d(original_time_rand, psth_eye_rand, kind='linear', fill_value='interpolate')
                    psth_eye_rand = f_eye_rand(target_time_rand)

                    f_pupilarea_rand = interpolate.interp1d(original_time_rand, psth_pupilarea_rand, kind='linear', fill_value='interpolate')
                    psth_pupilarea_rand = f_pupilarea_rand(target_time_rand)
                
                psth_whisker_random.append(psth_whisker_rand)
                psth_nose_random.append(psth_nose_rand)
                psth_eye_random.append(psth_eye_rand)
                psth_pupilarea_random.append(psth_pupilarea_rand)
        
            # Convert lists to numpy arrays
            psth_whisker_all = np.array(psth_whisker_all)
            psth_nose_all = np.array(psth_nose_all)
            psth_eye_all = np.array(psth_eye_all)
            psth_pupilarea_all = np.array(psth_pupilarea_all)
            psth_whisker_random = np.array(psth_whisker_random)
            psth_nose_random = np.array(psth_nose_random)
            psth_eye_random = np.array(psth_eye_random)
            psth_pupilarea_random = np.array(psth_pupilarea_random)

            # Compute mean speed values
            mean_psth_whisker = psth_whisker_all.mean(axis=0)
            mean_psth_nose = psth_nose_all.mean(axis=0)
            mean_psth_eye = psth_eye_all.mean(axis=0)
            mean_psth_pupilarea = psth_pupilarea_all.mean(axis=0)
            mean_psth_whisker_random = psth_whisker_random.mean(axis=0)
            mean_psth_nose_random = psth_nose_random.mean(axis=0)
            mean_psth_eye_random = psth_eye_random.mean(axis=0)
            mean_psth_pupilarea_random = psth_pupilarea_random.mean(axis=0)

            # Append to list
            mean_psth_whisker_all.append(mean_psth_whisker)
            mean_psth_nose_all.append(mean_psth_nose)
            mean_psth_eye_all.append(mean_psth_eye)
            mean_psth_pupilarea_all.append(mean_psth_pupilarea)
            mean_psth_whisker_random_all.append(mean_psth_whisker_random)
            mean_psth_nose_random_all.append(mean_psth_nose_random)
            mean_psth_eye_random_all.append(mean_psth_eye_random)
            mean_psth_pupilarea_random_all.append(mean_psth_pupilarea_random)

# Now all PSTHs are resampled to TARGET_SR, so we can use target_padding for analysis
padding = target_padding  # This is now consistent across all datasets

# Make a plot saving the post and pre avg. plot as line plot showing each mouse
pre_idx = np.arange(padding-10, padding) # pre-saccade 
post_idx = np.arange(padding, padding+10)
post_avg_random_whisker = np.mean(np.array(mean_psth_whisker_random_all)[:, post_idx], axis=1)
post_avg_random_nose = np.mean(np.array(mean_psth_nose_random_all)[:, post_idx], axis=1)
post_avg_whisker = np.mean(np.array(mean_psth_whisker_all)[:, post_idx], axis=1)
pre_avg_whisker = np.mean(np.array(mean_psth_whisker_all)[:, pre_idx], axis=1)
post_avg_nose = np.mean(np.array(mean_psth_nose_all)[:, post_idx], axis=1)
pre_avg_nose = np.mean(np.array(mean_psth_nose_all)[:, pre_idx], axis=1)
post_avg_pupilarea = np.mean(np.array(mean_psth_pupilarea_all)[:, post_idx], axis=1)
pre_avg_pupilarea = np.mean(np.array(mean_psth_pupilarea_all)[:, pre_idx], axis=1)
random_avg_pupilarea = np.nanmean(np.array(mean_psth_pupilarea_random_all)[:, post_idx], axis=1)

print(np.nanmean(random_avg_pupilarea), post_avg_nose.shape)
print(random_avg_pupilarea)

# Plot individual mouse lines
fig, ax = plt.subplots(1, 1, figsize=(4, 7))  
ax.plot([post_avg_random_whisker, pre_avg_whisker, post_avg_whisker], color='darkorange', alpha=0.25, lw=0.5)
ax.plot([post_avg_random_nose, pre_avg_nose, post_avg_nose], color='green', alpha=0.25, lw=0.5)
ax.plot([random_avg_pupilarea, pre_avg_pupilarea, post_avg_pupilarea], color='purple', alpha=0.25, lw=0.5)
# plot avg
ax.plot([np.mean(post_avg_random_whisker), np.mean(pre_avg_whisker), np.mean(post_avg_whisker)], 
        marker='o', color='darkorange', label='Whiskers', lw=2)
ax.plot([np.mean(post_avg_random_nose), np.mean(pre_avg_nose), np.mean(post_avg_nose)], 
        marker='o', color='green', label='Nose', lw=2) 
ax.plot([np.nanmean(random_avg_pupilarea), np.nanmean(pre_avg_pupilarea), np.nanmean(post_avg_pupilarea)],
        marker='o', color='purple', label='Pupil Area', lw=2) 
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Random', 'Pre-saccade', 'Post-saccade'])
ax.set_ylabel('Mean distance (px)')
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
whisker_data = [post_avg_random_whisker, pre_avg_whisker, post_avg_whisker]
nose_data = [post_avg_random_nose, pre_avg_nose, post_avg_nose]
pupilarea_data = [random_avg_pupilarea, pre_avg_pupilarea, post_avg_pupilarea]

print("Statistical Test Results:")

# Whisker comparisons
print("\nWhisker Comparisons:")
for (i, j) in combinations([0, 1, 2], 2):
    data1, data2 = whisker_data[i], whisker_data[j]
    stat, p = ttest_rel(data1, data2) if len(data1) == len(data2) else (np.nan, np.nan)
    test_type = "Paired t-test"
    
    # Plot significance
    y_max = max(np.mean(data1), np.mean(data2))
    height_offset = 0.01 + (i+j)*0.055  # Stagger heights
    if not np.isnan(p) and p<0.05:
        add_significance_bar(ax, i, j, y_max + height_offset*y_max, p, height_factor=0.01*y_max)
    
    print(f"{conditions[i]} vs {conditions[j]}: {test_type}, stat={stat:.3f}, p={p:.4f}")

# Nose comparisons
print("\nNose Comparisons:")
for (i, j) in combinations([0, 1, 2], 2):
    data1, data2 = nose_data[i], nose_data[j]
    stat, p = ttest_rel(data1, data2) if len(data1) == len(data2) else (np.nan, np.nan)
    test_type = "Paired t-test"
    
    # Plot significance - place below the whisker bars
    y_max = max(np.mean(data1), np.mean(data2))
    height_offset = (i+j)*0.075  # Negative offset for nose
    if not np.isnan(p) and p<0.05:
        add_significance_bar(ax, i, j, y_max + height_offset*y_max, p, height_factor=0.01*y_max)
    
    print(f"{conditions[i]} vs {conditions[j]}: {test_type}, stat={stat:.3f}, p={p:.4f}")

# Pupil area comparisons
print("\nPupil Area Comparisons:")
for (i, j) in combinations([0, 1, 2], 2):
    data1, data2 = pupilarea_data[i], pupilarea_data[j]
    stat, p = ttest_rel(data1, data2) if len(data1) == len(data2) else (np.nan, np.nan)
    test_type = "Paired t-test"
    
    # Plot significance - place below the nose bars
    y_max = max(np.mean(data1), np.mean(data2))
    height_offset = (i+j)*0.095  # More negative offset for pupil area
    if not np.isnan(p) and p<0.05:
        add_significance_bar(ax, i, j, y_max + height_offset*y_max, p, height_factor=0.01*y_max)
    
    print(f"{conditions[i]} vs {conditions[j]}: {test_type}, stat={stat:.3f}, p={p:.4f}")

plt.tight_layout()
plt.show()


# Get SEM values 
sem_psth_whisker_all = np.std(np.array(mean_psth_whisker_all), axis=0) / np.sqrt(len(mean_psth_whisker_all))
sem_psth_nose_all = np.std(np.array(mean_psth_nose_all), axis=0) / np.sqrt(len(mean_psth_nose_all))
sem_psth_eye_all = np.std(np.array(mean_psth_eye_all), axis=0) / np.sqrt(len(mean_psth_eye_all))
sem_psth_pupilarea_all = np.std(np.array(mean_psth_pupilarea_all), axis=0) / np.sqrt(len(mean_psth_pupilarea_all))
sem_psth_whisker_random_all = np.std(np.array(mean_psth_whisker_random_all), axis=0) / np.sqrt(len(mean_psth_whisker_random_all))
sem_psth_nose_random_all = np.std(np.array(mean_psth_nose_random_all), axis=0) / np.sqrt(len(mean_psth_nose_random_all))
sem_psth_eye_random_all = np.std(np.array(mean_psth_eye_random_all), axis=0) / np.sqrt(len(mean_psth_eye_random_all))
sem_psth_pupilarea_random_all = np.std(np.array(mean_psth_pupilarea_random_all), axis=0) / np.sqrt(len(mean_psth_pupilarea_random_all))

# Convert lists to numpy arrays and get mean values
mean_psth_whisker_all = np.mean(np.array(mean_psth_whisker_all), axis=0)
mean_psth_nose_all = np.mean(np.array(mean_psth_nose_all), axis=0)
mean_psth_eye_all = np.mean(np.array(mean_psth_eye_all), axis=0)
mean_psth_pupilarea_all = np.mean(np.array(mean_psth_pupilarea_all), axis=0)
mean_psth_whisker_random_all = np.mean(np.array(mean_psth_whisker_random_all), axis=0)
mean_psth_nose_random_all = np.mean(np.array(mean_psth_nose_random_all), axis=0)
mean_psth_eye_random_all = np.mean(np.array(mean_psth_eye_random_all), axis=0)
mean_psth_pupilarea_random_all = np.mean(np.array(mean_psth_pupilarea_random_all), axis=0)

# Smooth the traces for few frames
smooth_filter_len = 2
mean_psth_whisker_all = gaussian_filter(mean_psth_whisker_all, smooth_filter_len)
mean_psth_nose_all = gaussian_filter(mean_psth_nose_all, smooth_filter_len)
mean_psth_eye_all = gaussian_filter(mean_psth_eye_all, smooth_filter_len)
mean_psth_pupilarea_all = gaussian_filter(mean_psth_pupilarea_all, smooth_filter_len)
mean_psth_whisker_random_all = gaussian_filter(mean_psth_whisker_random_all, smooth_filter_len)
mean_psth_nose_random_all = gaussian_filter(mean_psth_nose_random_all, smooth_filter_len)
mean_psth_eye_random_all = gaussian_filter(mean_psth_eye_random_all, smooth_filter_len)
mean_psth_pupilarea_random_all = gaussian_filter(mean_psth_pupilarea_random_all, smooth_filter_len)

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(7, 5), sharey=True)

# Plot mean values with shaded error
line_lw = 1
ax[0].plot(mean_psth_whisker_all, label="Whiskers", lw=line_lw, color='darkorange')
ax[0].fill_between(np.arange(len(mean_psth_whisker_all)), mean_psth_whisker_all-sem_psth_whisker_all, mean_psth_whisker_all+sem_psth_whisker_all, alpha=0.2, color='darkorange')
ax[0].plot(mean_psth_nose_all, label="Nose", lw=line_lw, color='green')
ax[0].fill_between(np.arange(len(mean_psth_nose_all)), mean_psth_nose_all-sem_psth_nose_all, mean_psth_nose_all+sem_psth_nose_all, alpha=0.2, color='green')
ax[0].legend(loc='upper left', frameon=False)
ax[0].plot(mean_psth_pupilarea_all, label="Pupil Area", lw=line_lw, color='purple')
ax[0].fill_between(np.arange(len(mean_psth_pupilarea_all)), mean_psth_pupilarea_all-sem_psth_pupilarea_all, mean_psth_pupilarea_all+sem_psth_pupilarea_all, alpha=0.2, color='purple')

# Plot random mean values
ax[1].plot(mean_psth_whisker_random_all, label="Whiskers (Random)", lw=line_lw, color='darkorange')
ax[1].fill_between(np.arange(len(mean_psth_whisker_random_all)), mean_psth_whisker_random_all-sem_psth_whisker_random_all, mean_psth_whisker_random_all+sem_psth_whisker_random_all, alpha=0.2, color='darkorange')
ax[1].plot(mean_psth_nose_random_all, label="Nose (Random)", lw=line_lw, color='green')
ax[1].fill_between(np.arange(len(mean_psth_nose_random_all)), mean_psth_nose_random_all-sem_psth_nose_random_all, mean_psth_nose_random_all+sem_psth_nose_random_all, alpha=0.2, color='green')
ax[1].plot(mean_psth_pupilarea_random_all, label="Pupil Area (Random)", lw=line_lw, color='purple')
ax[1].fill_between(np.arange(len(mean_psth_pupilarea_random_all)), mean_psth_pupilarea_random_all-sem_psth_pupilarea_random_all, mean_psth_pupilarea_random_all+sem_psth_pupilarea_random_all, alpha=0.2, color='purple')

# Create time vector for x-axis (now using TARGET_SR)
n_samples = len(mean_psth_whisker_all)
time_vector = (np.arange(n_samples) - padding) / TARGET_SR  # in seconds

# Choose ticks based on time
tick_spacing_sec = 5  # ticks every x seconds
min_time = time_vector[0]
max_time = time_vector[-1]

xtick_labels = np.arange(np.ceil(min_time / tick_spacing_sec) * tick_spacing_sec,
                         np.floor(max_time / tick_spacing_sec) * tick_spacing_sec + tick_spacing_sec,
                         tick_spacing_sec)

# Map these labels back to data indices
xticks = (xtick_labels * TARGET_SR + padding).astype(int)
xtick_labels = np.round(xtick_labels, 1)

ax[0].set_xticks(xticks)
ax[0].set_xticklabels(xtick_labels)
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xtick_labels)

# Apply the ticks and labels to both subplots
for a in ax:
    a.set_xlim(0, len(mean_psth_whisker_all))
    
    # Add vertical line at center point (which will now correspond to 0 in the labels)
    ymin, ymax = np.min([ax[0].get_ylim()[0], ax[1].get_ylim()[0]]), np.max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]])
    a.vlines(padding, ymin=ymin, ymax=ymax, color='red', linestyle='--', lw=.8)

ax[0].set_xlabel('Time from saccade (s)')
ax[0].set_ylabel('Mean distance (px)')
ax[1].set_xlabel('Random window (s)')

# Remove spines
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

fig.suptitle("PSTH of Whisker, Nose and Eye Keypoints")
plt.tight_layout()
plt.show()

# Save data for plotting 
np.savez_compressed(f'{BASE_FOLDER}/data/saccade_data/psth_whisker_nose_eye.npz', 
                     mean_psth_whisker_all=mean_psth_whisker_all, 
                     mean_psth_nose_all=mean_psth_nose_all, 
                     mean_psth_eye_all=mean_psth_eye_all,
                     mean_psth_pupilarea_all=mean_psth_pupilarea_all,
                     mean_psth_whisker_random_all=mean_psth_whisker_random_all,
                     mean_psth_nose_random_all=mean_psth_nose_random_all,
                     mean_psth_eye_random_all=mean_psth_eye_random_all,
                     mean_psth_pupilarea_random_all=mean_psth_pupilarea_random_all,
                     sem_psth_whisker_all=sem_psth_whisker_all,
                     sem_psth_nose_all=sem_psth_nose_all,
                     sem_psth_eye_all=sem_psth_eye_all,
                     sem_psth_pupilarea_all=sem_psth_pupilarea_all,
                     sem_psth_whisker_random_all=sem_psth_whisker_random_all,
                     sem_psth_nose_random_all=sem_psth_nose_random_all,
                     sem_psth_eye_random_all=sem_psth_eye_random_all,
                     sem_psth_pupilarea_random_all=sem_psth_pupilarea_random_all,
                     post_avg_whisker=post_avg_whisker,
                     post_avg_nose=post_avg_nose,
                     post_avg_pupilarea=post_avg_pupilarea,
                     pre_avg_whisker=pre_avg_whisker,
                     pre_avg_nose=pre_avg_nose,
                     pre_avg_pupilarea=pre_avg_pupilarea,
                     post_avg_random_whisker=post_avg_random_whisker,
                     post_avg_random_nose=post_avg_random_nose,
                     random_avg_pupilarea=random_avg_pupilarea,
                     sampling_rate=TARGET_SR,  # Now using the target sampling rate
                     padding=padding)
print("File saved in ", f'{BASE_FOLDER}/data/saccade_data/')
