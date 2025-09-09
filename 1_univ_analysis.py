#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: remove PATHs and add logger

import os, glob, sys
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from tqdm import tqdm
from nilearn.image import index_img, load_img, new_img_like
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map


PATHs = {
    "bids": "/Volumes/XWang/projects/rsa-nf/VE1/dataset/3_prepro_data/bids" if sys.platform == "darwin" else "I:/projects/rsa-nf/VE1/dataset/3_prepro_data/bids", 
    "event": "../resources/task-events.tsv",
    "voi_mask": "../resources/voi_mask_mni.nii.gz",
    "save_stat": "../results/statistics",
    "save_fig": "../results/figures"
    }




def load_data(sub_id, task, run_id):
    func_img = os.path.join(PATHs["bids"], "derivatives", f"sub-{sub_id:02d}", "func",
                            f"sub-{sub_id:02d}_task-{task}_run-{run_id:02d}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    mask_img = os.path.join(PATHs["bids"], "derivatives", f"sub-{sub_id:02d}", "func",
                            f"sub-{sub_id:02d}_task-{task}_run-{run_id:02d}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz")
    events = pd.read_csv(PATHs["event"], sep = '\t')[["onset", "duration", "trial_type"]]
    return func_img, mask_img, events




# build confounds derived from fmriprep (motion, physiological, motion outlier)
# discard the first and last a few volumes
def build_confounds(sub_id, task, run_id, trim_volumes = 5):
    confounds_tsv = os.path.join(PATHs["bids"], "derivatives", f"sub-{sub_id:02d}", "func",
                                 f"sub-{sub_id:02d}_task-{task}_run-{run_id:02d}_desc-confounds_timeseries.tsv")
    confounds = pd.read_csv(confounds_tsv, sep = '\t')
    if task == "NF": confounds = confounds.iloc[20:].reset_index(drop = True)
    # Confounds to remove the first and last a few volumes
    discard_volumes = np.ones(confounds.shape[0])
    discard_volumes[trim_volumes: -trim_volumes] = 0
    discard_volumes = pd.DataFrame(discard_volumes, columns = ["discard_volumes"])
    # Motion confounds
    motion_confounds = confounds[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]].fillna(0)
    # Physiological confounds
    a_comp_cor = [f'a_comp_cor_{i:02d}' for i in range(1, 6)]
    physio_confounds = confounds[["csf", "white_matter"] + a_comp_cor].fillna(0)
    # Motion outlier confounds
    motion_outlier_cols = [col for col in confounds.columns if "motion_outlier" in col]
    if motion_outlier_cols:
        outlier_confounds = confounds[motion_outlier_cols]
        confounds_df = pd.concat([motion_confounds, physio_confounds, outlier_confounds, discard_volumes], axis = 1)
    else:
        confounds_df = pd.concat([motion_confounds, physio_confounds, discard_volumes], axis = 1)
    return confounds_df




# cluster-level correction using Monte Carlo Simulations
def cluster_correction(stat_img, voxel_thresh, cluster_thresh = 0.05, num_simulations = 1000):
    stat_data = stat_img.get_fdata()
    binary_map = np.abs(stat_data) > voxel_thresh
    # label and compute the size of connected components (clusters) in the binary map
    labeled_clusters, num_clusters = ndimage.label(binary_map)
    cluster_sizes = ndimage.sum(binary_map, labeled_clusters, range(1, num_clusters + 1))
    # compute cluster-size threshold using Monte Carlo Simulations
    simulated_max_cluster_sizes = []
    for _ in tqdm(range(num_simulations), desc = "Monte Carlo Simulation:"):
        random_noise = np.random.normal(0, 1, stat_data.shape)      # Simulating null distribution
        random_binary_map = np.abs(random_noise) > voxel_thresh     # Apply same threshold
        random_labeled_clusters, _ = ndimage.label(random_binary_map)
        random_cluster_sizes = ndimage.sum(random_binary_map, random_labeled_clusters, range(1, num_clusters + 1))
        if len(random_cluster_sizes) > 0: simulated_max_cluster_sizes.append(np.max(random_cluster_sizes))
        else: simulated_max_cluster_sizes.append(0)                 # Handle cases where no clusters exist
    cluster_size_threshold = np.percentile(simulated_max_cluster_sizes, int(100 * (1 - cluster_thresh)))
    print(f"Cluster-size threshold for p < {cluster_thresh}: {cluster_size_threshold} voxels")
    # apply cluster-size threshold to the original data
    significant_clusters = np.where(cluster_sizes >= cluster_size_threshold)[0] + 1
    corrected_data = np.isin(labeled_clusters, significant_clusters) * stat_data
    corrected_img = new_img_like(stat_img, corrected_data)
    return corrected_img




# plot contrast map using nilearn
def plot_map(map_data, fig_path, cmap = "cold_hot", vmax = 10):
    plot_stat_map(map_data, display_mode = "z", cut_coords = [-16, 0, 16, 40], vmax = vmax, cmap = cmap, colorbar = False)
    fig = plt.gcf()
    for text in fig.findobj(plt.Text): text.set_fontsize(25)
    plt.savefig(fig_path, dpi = 300)
    plt.close()




# first-level analysis for each run (contrast: task vs rest)
def first_level_analysis(sub_id, task, run_id, voxel_thresh = 2.576, cluster_thresh = 0.05):
    print(f"Running first-level analysis for sub-{sub_id:02d}_task-{task}_run-{run_id:02d}")
    stat_path = os.path.join(PATHs["save_stat"], f'sub-{sub_id:02d}_task-{task}_run-{run_id:02d}_task-vs-rest.nii.gz')
    if os.path.exists(stat_path): return
    func_img, mask_img, events = load_data(sub_id, task, run_id)
    if not os.path.exists(func_img): return                             # Skip if no data available
    if task == "NF": func_img = index_img(func_img, slice(20, None))    # Skip first 20 volumes for NF task
    confounds = build_confounds(sub_id, task, run_id)
    # fit first-level model
    first_level_model = FirstLevelModel(
        t_r = 1.0, slice_time_ref = 0.5, hrf_model = "spm", drift_model = 'cosine',
        high_pass = 1/128, smoothing_fwhm = 6.0, noise_model = 'ar1', standardize = True, mask_img = mask_img
    )
    first_level_model.fit(func_img, events = events, confounds = confounds)
    # build contrast matrix (task vs rest)
    design_matrix = first_level_model.design_matrices_[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = {column: contrast_matrix[i] for i, column in enumerate(design_matrix.columns)}
    task_vs_rest_contrast = basic_contrasts["task"] - basic_contrasts["rest"]
    # compute contrast map and apply cluster-level correction
    contrast_map = first_level_model.compute_contrast(task_vs_rest_contrast, output_type = 'z_score')
    corrected_map = cluster_correction(contrast_map, voxel_thresh = voxel_thresh, cluster_thresh = cluster_thresh)
    corrected_map.to_filename(stat_path)




# second-level analysis for each run
# uncorrected voxel-wise threshold (p < 0.05), cluster-level threshold (p < 0.05)
def second_level_analysis(task, run_id, total_sub_num = 27, voxel_thresh = 2.704, cluster_thresh = 0.05, desc = ""):
    print(f"Running second-level analysis for task-{task}_run-{run_id:02d}")
    stat_path = os.path.join(PATHs["save_stat"], f'group_task-{task}_run-{run_id:02d}_task-vs-rest{desc}.nii.gz')
    fig_path = os.path.join(PATHs["save_fig"], f'group_task-{task}_run-{run_id:02d}_task-vs-rest{desc}.png')
    if not os.path.exists(stat_path):
        # first-level analysis for each subject
        for sub_id in range(1, total_sub_num+1): first_level_analysis(sub_id, task, run_id)
        # load group contrast maps according to the task and run
        contrast_files = glob.glob(os.path.join(PATHs["save_stat"], f'sub*task-{task}_run-{run_id:02d}_task-vs-rest.nii.gz'))
        if task == "NF":        # special case: run-02 from sub-01 is for sadness instead of joyfulrelaxation
            if run_id < 3: contrast_files = [f for f in contrast_files if not "sub-01" in f]
            else: contrast_files.append(os.path.join(PATHs["save_stat"], 'sub-01_task-NF_run-02_task-vs-rest.nii.gz'))
        elif len(desc) > 0:     # exclude some subjects in localizer run for comparison with NF run 
            if run_id == 1: excluded_subs = ["sub-01", "sub-13", "sub-24"]
            else: excluded_subs = ["sub-04", "sub-05", "sub-06", "sub-07", "sub-13", "sub-15", "sub-18", \
                "sub-20", "sub-22", "sub-23", "sub-24", "sub-26", "sub-27"]
            contrast_files = [f for f in contrast_files if not any(sub in f for sub in excluded_subs)]
        contrast_imgs = [nib.load(f) for f in contrast_files]
        # fit second-level model
        design_matrix = pd.DataFrame(np.ones((len(contrast_imgs), 1)), columns = ['Intercept'])
        second_level_model = SecondLevelModel(mask_img = load_img(PATHs["voi_mask"]))
        second_level_model.fit(contrast_imgs, design_matrix = design_matrix)
        # compute group contrast map and apply cluster-level correction
        group_map = second_level_model.compute_contrast(output_type = 'stat')
        group_map = cluster_correction(group_map, voxel_thresh = voxel_thresh, cluster_thresh = cluster_thresh)
        group_map.to_filename(stat_path)
    # plot group contrast map
    group_map = nib.load(stat_path)
    plot_map(group_map, fig_path)




if __name__ == "__main__":
    # Localizer data analysis (N = 27)
    for run_id in range(1, 5):
        second_level_analysis("localizer", run_id)

    # NF data analysis (Joyfulrelaxation, N = 24)
    second_level_analysis("localizer", run_id = 1, desc = "_compare-NF")
    for run_id in range(1, 3):
        second_level_analysis("NF", run_id)

    # NF data analysis (Sadness, N = 14)
    second_level_analysis("localizer", run_id = 2, desc = "_compare-NF")
    second_level_analysis("NF", run_id = 3)
