import numpy as np
import matplotlib.pyplot as plt
import os, sys
import nonlinear_modelv3
import helper_functions
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import pupil_dark_helpers as pdh
from scipy.io import loadmat


BASE_FOLDER = os.getcwd().replace('analysis/code', '') # base folder for all data
print("Base folder: ", BASE_FOLDER) # this should be parent folder called eye-vs-faces

dark_dbs = pdh.carbachol_in_dark() 
light_random_dbs = pdh.get_light_random_info()
light_static_dbs = pdh.get_light_static_info()

dbs = [dark_dbs, light_random_dbs, light_static_dbs]
category = ['dark', 'light_random', 'light_static']

frequencies = []
condition = []
durations = []
saccade_directions = []
saccade_direction_angles = []
saccade_distances = []
saccade_durations = []
beta_x, beta_y = [], []
peak_velocities = []
saccade_amplitudes = []

saccade_times = {}
blink_times = {}

for db_idx, db in enumerate(dbs):
    saccade_times[category[db_idx]] = {} 
    blink_times[category[db_idx]] = {}
    # Get mouse list
    mouse_list = np.unique([entry['mname'] for entry in db])
    print('Mouse list: ', mouse_list, category[db_idx])
    for mname in mouse_list:
        if mname == 'TX150':  # skip this mouse due to different sampling rate
            continue
        print('>>>>>>Processing mouse: ', mname)
        # Get datexp 
        datexp = [entry['datexp'] for entry in db if entry['mname'] == mname][0]
        saccade_times[category[db_idx]][mname] = {} 
        blink_times[category[db_idx]][mname] = {}
        # Load data
        _, _, _, _, _, _, _, npz_data = nonlinear_modelv3.load_data(f'{BASE_FOLDER}/data/npz_data/', mname, datexp)

        # segment the data into blocks
        block_frames = npz_data['cam_frames_list']
        
        # Get number of blocks in the recording
        blocks = list(npz_data['tcam'].item())
        num_blocks = len(blocks)
        
        if 'light' in category[db_idx]:
            block_idxs = np.arange(1)
        else:
            block_idxs = np.arange(1, num_blocks)

        total_saccades = 0
        total_time = 0
        for blk_idx in block_idxs: 
            blk = blocks[blk_idx].split('blk')[1]
            if category[db_idx] == 'dark' and blk == '1': # no carcabachol in first block
                continue
            print('block: ', blk)

            # Get pupil timestamp
            pupil_timestamp = npz_data['tcam'].item()[list(npz_data['tcam'].item())[blk_idx]]
            pupil_x = helper_functions.segment_data(npz_data["pupil_x_small"], block_frames, blk_idx)
            pupil_y = helper_functions.segment_data(npz_data["pupil_y_small"] , block_frames, blk_idx)
            blink = helper_functions.segment_data(npz_data["blink_small"], block_frames, blk_idx)
            pupil_xy = np.vstack((pupil_x, pupil_y)).T
            polar_x = helper_functions.segment_data(npz_data["pupil_x_polar"], block_frames, blk_idx)
            polar_y = helper_functions.segment_data(npz_data["pupil_y_polar"], block_frames, blk_idx)

            # Camera view for dark recordings is flipped; correct polar x values
            if (category[db_idx] == 'dark') or (mname in ['FXa10', 'FXa8', 'FXa9', 'FXa1']):  # gcamp8 and dark recordings have view nose(left)-to-ear (right)
                polar_x = np.mod(np.pi - polar_x, 2 * np.pi)

            # Baseline correction for blink artifacts
            blink_baselined_threshold = float([entry['blink_baselined_threshold'] for entry in db if entry['mname'] == mname and entry['blk']==blk][0])
            blink_baselined = gaussian_filter(blink, 10)
            blink_baselined = minimum_filter1d(blink_baselined, 5)
            blink_baselined = maximum_filter1d(blink_baselined, 5)
            blink_baselined = blink - blink_baselined
            # Detect blinks
            Sblk, Eblk = helper_functions.detect_blinks(blink_baselined, pupil_timestamp, blink_baselined_threshold)
            blink_times[category[db_idx]][mname][blk] = np.array(Eblk)

            # Filter out high frequency noise
            cutoff_freq = 1.5  # Adjust this value depending on the movement frequency you want to filter out (in Hz)
            cam_time_min = np.min(pupil_timestamp)
            cam_time = pupil_timestamp.copy()
            cam_time -= cam_time_min
            cam_time *= 24 * 60 * 60  # Convert to seconds
            sampling_rate = 1 / np.diff(cam_time).mean()

            # Get velocity of pupil
            pupil_x_diff = np.diff(pupil_x, prepend=pupil_x[0])
            pupil_y_diff = np.diff(pupil_y, prepend=pupil_y[0])
            pupil_velocity = np.sqrt(pupil_x_diff**2 + pupil_y_diff**2) * sampling_rate  # Convert to pixels per second

            # Get saccade indices from finetuned labels of the model
            if 'dark' in category[db_idx] or mname in ['FXa10', 'FXa8', 'FXa9', 'FXa1']:
                saccade_labels_filename = f"{BASE_FOLDER}/data/saccade_labels/{mname}_{datexp}_{blk}_saccade_labels.mat"
            else: # other light recordings
                saccade_labels_filename = f"{BASE_FOLDER}/data/saccade_labels/{mname}_{datexp}_finetuned_model_labels.mat" 
            if not os.path.exists(saccade_labels_filename):
                print(f"Labels file not found for {mname} on {datexp}. Skipping block {blk}.")
                continue
            saccade_labels = loadmat(saccade_labels_filename)['eye_movement_data']['Saccade'][0,0].squeeze()
            sac_start_idx = np.where(np.diff(saccade_labels) == 1)[0] + 1
            sac_end_idx = np.where(np.diff(saccade_labels) == -1)[0] + 1
            print("num saccades: ", len(sac_start_idx))

            # ~~~~~~~~~~~~~~~~~~ Get all saccade related details ~~~~~~~~~~~~~~~~~~
            saccade_info = []
            for sac in zip(sac_start_idx, sac_end_idx):
                start_idx, end_idx = sac
                start_time = pupil_timestamp[start_idx]
                end_time = pupil_timestamp[end_idx]
                start_x = pupil_x[start_idx]
                start_y = pupil_y[start_idx]
                end_x = pupil_x[end_idx]
                end_y = pupil_y[end_idx]
                max_velocity_idx = np.argmax(pupil_velocity[start_idx:end_idx]) + start_idx
                beta_x.append(polar_x[end_idx]-polar_x[start_idx])
                beta_y.append(polar_y[end_idx]-polar_y[start_idx])
                if polar_x[start_idx] < polar_x[end_idx]:
                    saccade_directions.append('Nasal') # Rightward in view is nasal in mouse
                else:
                    saccade_directions.append('Temporal') # Leftward in view is temporal in mouse
                angle = np.arctan2(end_y - start_y, end_x - start_x) * 180 / np.pi
                distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                saccade_distances.append(distance)
                saccade_direction_angles.append(angle)
                saccade_durations.append((end_time - start_time) * 24 * 60 * 60 * 1000)  # Convert to milliseconds

                saccade_info.append([start_idx, end_idx, start_time, end_time, start_x, start_y, end_x, end_y, angle, saccade_directions[-1], max_velocity_idx])

            saccade_times[category[db_idx]][mname][blk] = saccade_info

            # ~~~~~~~~~~~~~~~~~~ Plot saccades aligned to max velocity in start:end idx range ~~~~~~~~~~~~~~~~~
            max_velocity_idx = []
            for start_idx, end_idx in zip(sac_start_idx, sac_end_idx):
                # get index of max velocity in the range start_idx:end_idx
                max_velocity_idx.append(np.argmax(pupil_velocity[start_idx:end_idx]) + start_idx)
                # store in peak_velocities and amplitudes
                x_start, y_start = pupil_x[start_idx], pupil_y[start_idx]
                x_end, y_end = pupil_x[end_idx], pupil_y[end_idx]
                amplitude = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2) # euclidean distance
                saccade_amplitudes.append(amplitude)
                peak_velocities.append(np.max(pupil_velocity[start_idx:end_idx]))

            # Get saccade statistics
            tstart = pupil_timestamp[0] * 24 * 60 * 60
            tend = pupil_timestamp[-1] * 24 * 60 * 60 
            total_saccades += len(sac_start_idx)
            total_time += (tend - tstart)
            print("sampling rate: ", sampling_rate)
            
        # compute frequency in saccades per minute
        saccade_frequency = total_saccades / total_time  # saccades per second
        frequencies.append(saccade_frequency * 60) # convert to per minute
        condition.append(category[db_idx])
        print("frequency (saccades per min): ", saccade_frequency * 60)
        print("condition: ", category[db_idx])

saccade_info = np.array(saccade_info)

# Plot saccade durations histogram
plt.figure(figsize=(7, 4))
plt.hist(saccade_durations, bins=20, alpha=0.7, color='grey')
plt.title('Saccade Duration Distribution')
plt.xlabel('Saccade Duration (ms)')
plt.ylabel('Count')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

# Calculate the angles from beta_x and beta_y
beta_x = np.array(beta_x)
beta_y = np.array(beta_y)
# Calculate angles in radians
angles_rad = np.arctan2(beta_y, beta_x)  # This gives angles in [-π, π]

# Now use your existing polar plot code
bins = np.linspace(0, 2 * np.pi, 20)
counts, bin_edges = np.histogram(angles_rad, bins=bins)
probabilities = counts / np.sum(counts)  

# Create polar plot
plt.figure(figsize=(4, 4))
ax = plt.subplot(111, projection='polar')
bars = ax.bar(bin_edges[:-1], probabilities, width=np.diff(bin_edges), color='grey', edgecolor='k', alpha=0.7, align='center')

# Add probability values on top of each bar
for angle, prob, bar in zip(bin_edges[:-1], probabilities, bars):
    if prob > 0.2:
        height = bar.get_height()
        ax.text(
            angle,  # Angle position
            height + 0.03,  # Slightly above the bar
            f"{prob:.2f}",  # Format probability to 2 decimal places
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            fontsize=12,  # Adjust font size if needed
        )

ax.set_xticks(np.pi / 2 * np.array([0, 1, 2, 3]))  # 0° (right), 90° (up), 180° (left), 270° (down)
ax.set_xticklabels(['Nasal', 'Dorsal', 'Temporal', 'Ventral'], fontsize=10)
ax.set_yticklabels([])  # Remove radial axis numbers
ax.set_title("Saccade Direction Probabilities")
plt.show()

# Plot saccade angular displacement
plt.figure(figsize=(10, 6))
# Compute angular displacement (handling wrapping)
angular_displacement = np.diff(angles_rad)
angular_displacement = (angular_displacement + np.pi) % (2 * np.pi) - np.pi
# convert to degrees
angular_displacement = np.rad2deg(angular_displacement)
plt.hist(angular_displacement, bins=20, alpha=0.7, color='orange', edgecolor='black')
plt.title('Saccade Angular Displacement Distribution')
plt.xlabel('Angular Displacement (radians)')
plt.ylabel('Count')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

# Plot peak velocities vs. amplitudes scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(peak_velocities, saccade_amplitudes, alpha=0.5, color='blue')
plt.title('Peak Velocities vs. Saccade Amplitudes')
plt.xlabel('Peak Velocity (pixels/frame)')
plt.ylabel('Saccade Amplitude (pixels)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

# Save all data used to plot saccade direction probabilities
saccade_data = {
    'saccade_directions': saccade_directions,
    'beta_x': beta_x,
    'beta_y': beta_y,
    'saccade_distances': saccade_distances,
    'saccade_durations': saccade_durations,
    'frequencies': frequencies,
    'condition': condition,
    'peak_velocities': peak_velocities,
    'saccade_amplitudes': saccade_amplitudes,
}
# Save saccade data
np.savez(f'{BASE_FOLDER}/data/figure_data/saccade_data/saccade_data.npz', **saccade_data)
np.save(f'{BASE_FOLDER}/data/figure_data/saccade_data/saccade_times.npy', saccade_times)
np.save(f'{BASE_FOLDER}/data/figure_data/saccade_data/blink_times.npy', blink_times)
