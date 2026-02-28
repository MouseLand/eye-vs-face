---

# Figure Generation Notebooks

This repository contains Jupyter notebooks used to generate processed data, model outputs, and visualizations for the main and supplementary figures. Estimated time for each notebook is listed in each section accounting for the time required to run the entire dataset.

To reproduce the results, run the notebooks in the order listed below.

---

## Execution Order

### 1. `generate_svds.ipynb` (optional) [~8-10 hours]

**Generates:**

* Figure 1 data
* Figure 2 data

**Description:**
Computes PCA of behavioral videos using Facemap code within the notebook. Produces dimensionality-reduced features used as inputs for downstream model training and analysis.

**Output:**

* Saved results in `paper/data/mouse_data/`

---

### 2. `geom_to_deg_saccade.ipynb` (optional)  [~1 hour]

**Generates:**

* Figure 1 data
* Figure 2 data

**Description:**
Converts geometric pupil position coordinates (x, y) into degrees. These processed measurements are used for visualization and model training.

**Output:**

* Saved results in `paper/data/mouse_data/`

---

### 3. `saccade_detection.ipynb` (optional)  [~1 hour]

**Purpose:**
Initial saccade detection and label generation.

**Description:**
Uses the pretrained model (Andersson weights) to generate initial saccade predictions from behavioral videos. These predictions serve as provisional labels and should be manually refined in Facemap before being used for fine-tuning.

**Workflow:**

1. Run the pretrained model to generate saccade predictions.
2. Export predicted labels.
3. Manually refine labels in Facemap.
4. Use refined labels for downstream model training (`run_predictions.ipynb`).

**Output:**

* Exported files for refinement in Facemap
* Saved results in `paper/data/saccade_labels/`

---

### 4. `run_predictions.ipynb` [~4-5 hours]

**Generates:**

* Figure 1 results
* Figure 2 results

**Description:**
Trains and/or runs the fine-tuned model using refined saccade labels. Evaluates performance across experimental conditions and produces prediction outputs used for Figures 1 and 2.

**Output:**

* Saved results in `paper/data/figure_data/`

---

### 5. `tuning_curves.ipynb` [< 20 mins]

**Generates:**

* Figure 1 results

**Description:**
Computes and visualizes tuning curves for static image and random image conditions.

**Output:**

* Tuning curve plots
* Saved results in `paper/data/tuning_curve_data/`

---

### 6. `keypoints_prediction.ipynb` [~1-2 hours]

**Generates:**

* Figure 3 results

**Description:**
Predicts keypoints and pupil position from another set of keypoints or pupil measurements. Produces panels used in Figure 3.

**Output:**

* Figure 3 panels
* Saved results in `paper/data/saccade_data/`

---

### 7. `saccade_video.ipynb` [< 20 mins]

**Generates:**

* Supplementary Video

**Description:**
Creates a video visualization of detected saccades aligned with behavioral data.

**Output:**

* Exported video file (e.g., `.mp4`) saved in `paper/data/saccade_data/`

---

Although some of the notebooks can be run independently, running the notebooks in this order ensures that all required intermediate files are generated before downstream analyses and figure production. 

Add the following section to the `analysis/README.md` after the notebooks instructions.

---

# Code Modules (`analysis/code/`)

The `code/` directory contains reusable Python modules that support data processing, model training, and figure generation. Most of these modules are imported directly by the notebooks in `analysis/notebooks/`.

## Module Overview

### Modules Used by Notebooks [< 30 mins]

These files are imported automatically within the notebooks and do not need to be run directly:

* `nonlinear_modelv3.py` — Model definitions, architecture, and training routines
* `helper_functions.py` — General utility functions
* `pupil_dark_helpers*.py` — Recording details

These modules must remain in the `analysis/code/` directory so they can be imported correctly. Ensure that the `analysis` directory is in your Python path when running notebooks.

---

Modules That Must Be Run Directly:

The following scripts are **not** called automatically by the notebooks and must be executed manually in the following order to generate required intermediate datasets:

### 1. `get_saccades.py`

**Purpose:**
Compiles dataset-wide saccade information across recordings.

**Output:**

* Aggregated saccade data saved to disk
* Used for downstream analyses and figure generation

Run this script before analyses that require compiled saccade datasets.

---

### 2. `get_keypoints_psth.py`

**Purpose:**
Computes keypoint peri-saccade time histograms (PSTHs) aligned to saccade events. Used in Figure 3 analyses.

**Output:**

* Keypoint-aligned PSTH data saved for downstream plotting

---

### 3. `get_running_psth.py`

**Purpose:**
Computes running-aligned peri-saccade time histograms (PSTHs) aligned to saccade events. Used in Figure 3 analyses.

**Output:**

* Running-aligned PSTH data saved for downstream plotting

---

## How to Run Standalone Scripts

From the `analysis/` directory:

```bash
cd code
python get_saccades.py
python get_keypoints_psth.py
python get_running_psth.py
```

If you get an error, ensure required data files are already generated from earlier notebook steps before running these scripts.

---

## Summary

* `analysis/notebooks/` → Main execution pipeline for figures
* `analysis/code/` → Reusable modules and standalone preprocessing scripts
* Only three scripts require direct execution:

  * `get_saccades.py`
  * `get_keypoints_psth.py`
  * `get_running_psth.py`

All other modules are automatically used within the notebooks.

