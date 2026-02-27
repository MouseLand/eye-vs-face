---

# Figure Generation Notebooks

This repository contains Jupyter notebooks used to generate processed data, model outputs, and visualizations for the main and supplementary figures.

To reproduce the results, run the notebooks in the order listed below.

---

# Execution Order

## 1. `generate_svds.ipynb`

**Generates:**

* Figure 1 data
* Figure 2 data

**Description:**
Computes PCA of behavioral videos using Facemap code within the notebook. Produces dimensionality-reduced features used as inputs for downstream model training and analysis.

**Output:**

* Saved results in `paper/data/mouse_data/`

---

## 2. `geom_to_deg_saccade.ipynb` (optional)

**Generates:**

* Figure 1 data
* Figure 2 data

**Description:**
Converts geometric pupil position coordinates (x, y) into degrees. These processed measurements are used for visualization and model training.

**Output:**

* Saved results in `paper/data/mouse_data/`

---

## 3. `saccade_detection.ipynb`

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

## 4. `run_predictions.ipynb`

**Generates:**

* Figure 1 results
* Figure 2 results

**Description:**
Trains and/or runs the fine-tuned model using refined saccade labels. Evaluates performance across experimental conditions and produces prediction outputs used for Figures 1 and 2.

**Output:**

* Saved results in `paper/data/figure_data/`

---

## 5. `tuning_curves.ipynb`

**Generates:**

* Figure 1 results

**Description:**
Computes and visualizes tuning curves for static image and random image conditions.

**Output:**

* Tuning curve plots
* Saved results in `paper/data/tuning_curve_data/`

---

## 6. `keypoints_prediction.ipynb`

**Generates:**

* Figure 3 results

**Description:**
Predicts keypoints and pupil position from another set of keypoints or pupil measurements. Produces panels used in Figure 3.

**Output:**

* Figure 3 panels
* Saved results in `paper/data/saccade_data/`

---

## 7. `saccade_video.ipynb`

**Generates:**

* Supplementary Video

**Description:**
Creates a video visualization of detected saccades aligned with behavioral data.

**Output:**

* Exported video file (e.g., `.mp4`) saved in `paper/data/saccade_data/`

---

Although some of the notebooks can be run independently, running the notebooks in this order ensures that all required intermediate files are generated before downstream analyses and figure production. 
