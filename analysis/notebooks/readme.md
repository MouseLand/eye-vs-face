---

# Figure Generation Notebooks

This repository contains Jupyter notebooks used to generate output data used to make the main and supplementary figures. Each notebook corresponds to specific figures in the manuscript.

## Notebook Overview

### 1. `generate_svds.ipynb`

**Generates:**

* Figure 1 data
* Figure 2 data

**Description:**
Computes PCA of behavioral videos using the Facemap software code in notebook and produces the input data used for training models for Figure 1 and 2. 

**Output:**

* Saved results in paper/data/mouse_data.

---

### 2. `geom_to_deg_saccade.ipynb`

**Generates:**

* Figure 1 and 2 data

**Description:**
Converts geometric measurements (x, y) pupil position coordinates into degrees for visualizations and model training in figure 1 and 2.

**Output:**

* Saved results in paper/data/mouse_data.

---

### 3. `run_predictions.ipynb`

**Generates:**

* Figure 1 results
* Figure 2 results

**Description:**
Runs model predictions and evaluates performance across experimental conditions.

**Output:**

* Saved results in paper/data/figure_data.

---

### 4. `tuning_curves.ipynb`

**Generates:**

* Figure 1 results

**Description:**
Computes and visualizes tuning curves for static image and random images conditions.

**Output:**

* Tuning curve plots
* Saved results in paper/data/tuning_curve_data.

---

### 5. `keypoints_prediction.ipynb`

**Generates:**

* Figure 3 results

**Description:**
Code script used to predict keypoints and pupil position from another set of keypoints or pupil position.

**Output:**

* Figure 3 panels
* Saved results in paper/data/saccade_data.

---

### 6. `saccade_video.ipynb`

**Generates:**

* Supplementary Video

**Description:**
Creates a video visualization of saccades aligned with relevant behavioral data.

**Output:**

* Exported video file (e.g., `.mp4`) saved in paper/data/saccade_data.

---
