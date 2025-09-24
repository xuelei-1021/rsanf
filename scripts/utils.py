#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns
import scipy.stats as stats

from nilearn.image import new_img_like, load_img
from nilearn.plotting import plot_stat_map
from scipy import ndimage
from tqdm import tqdm


ANGLEs = [45, 135, 225, 315]
CONDITIONs = ["Anger", "Enthusiasm", "Joyfulrelaxation", "Sadness"]


AAL3_label2name = {
    "Frontal_Inf_Orb_2": "IFG (orbital)",
    "Frontal_Inf_Tri": "IFG (triangular)",
    "Frontal_Med_Orb": "SFG (medial orbital)",
    "Frontal_Sup_2": "SFG (dorsolateral)",
    "OFClat": "OFC (lateral)",
    "OFCpost": "OFC (posterior)",
    "Frontal_Sup_Medial": "SFG (medial)",
    "Frontal_Mid_2": "MFG",
    "ACC_sup": "ACC (subgenual)",
    "Rectus": "Rectus",
    "Frontal_Inf_Oper": "IFG (opercular)",
    "ParaHippocampal": "Parahippocampal",
    "ACC_pre": "ACC (pregenual)",
    "OFCant": "OFC (anterior)",
    "Insula": "Insula",
    "Hippocampus": "Hippocampus",
    "ACC_sub": "ACC (subgenual)",
    "OFCmed": "OFC (medial)",
    "Amygdala": "Amygdala",
    "Olfactory": "Olfactory",
}




def set_log(filename, path, stream_handler = True, level = logging.INFO):
	os.makedirs(path, exist_ok=True)
	logFormatter = logging.Formatter("%(asctime)s   %(levelname)s    %(message)s")
	rootLogger = logging.getLogger(filename)
	if not rootLogger.handlers:
		fileHandler = logging.FileHandler(os.path.join(path, filename))
		fileHandler.setFormatter(logFormatter)
		rootLogger.addHandler(fileHandler)
		if stream_handler:
			consoleHandler = logging.StreamHandler()
			consoleHandler.setFormatter(logFormatter)
			rootLogger.addHandler(consoleHandler)
		rootLogger.setLevel(level)
	return rootLogger




# cluster-level correction using Monte Carlo Simulations
def cluster_correction(stat_img, voxel_thresh, cluster_thresh = 0.05, num_simulations = 1000, logger = None):
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
    if logger: logger.info(f"Cluster-size threshold for p < {cluster_thresh}: {cluster_size_threshold} voxels")
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
    plt.show()
    # plt.savefig(fig_path, dpi = 300)
    # plt.close()




def load_aal3(filename):
    aal3_img = load_img(filename + ".gz")
    aal3_data = aal3_img.get_fdata()
    with open(filename + ".txt", "r") as f:
        aal3_labels = [line.strip().split(' ')[:2] for line in f.readlines()]
    aal3_id2label = {int(idx): label for idx, label in aal3_labels}
    aal3_label2id = {label: int(idx) for idx, label in aal3_labels}
    return aal3_img, aal3_data, aal3_labels, aal3_id2label, aal3_label2id




def read_voi_file(voi_file):
    coordinates = []
    with open(voi_file, 'r') as file:
        in_voi_section = False
        for line in file.readlines():
            line = line.strip()
            if line.startswith("NrOfVoxels:"):
                in_voi_section = True
                continue
            if in_voi_section:
                if line == "":
                    in_voi_section = False
                else:
                    coords = list(map(int, line.split()))
                    coordinates.append(coords)
    return coordinates




def load_voi_file(sub, aal3_img):
    voi_file = os.path.join(sub, "TBV_Files_Template_10blocks/rtRSA_output/selected_voxels.voi")
    plot_coords = []
    coords = read_voi_file(voi_file)
    for coord in coords:
        coord_tuple = tuple(coord)
        plot_coords.append(coord_tuple)
    voxel_coords = np.round(nib.affines.apply_affine(np.linalg.inv(aal3_img.affine), np.array(plot_coords))).astype(int)
    voxel_coords = np.unique(voxel_coords, axis = 0)
    return voxel_coords




def find_overlap(vec1, vec2, overlap_num = 10):
    scores = {val: idx for idx, val in enumerate(vec1)}
    for idx, val in enumerate(vec2):
        scores[val] += idx
    score_keys = list(scores.keys())
    score_keys.sort(key = lambda x: scores[x])
    return score_keys[:overlap_num]




def compute_RDM_variability(corrs):
    z_corrs = fisher_z_transform(corrs)
    z_mean = np.mean(z_corrs, axis = 0)
    z_std = np.std(z_corrs, axis = 0)
    z_low = z_mean - 1.96 * z_std / np.sqrt(z_corrs.shape[0])
    z_high = z_mean + 1.96 * z_std / np.sqrt(z_corrs.shape[0])
    RDM_low = 1 - inverse_fisher_z_transform(z_high)
    RDM_high = 1 - inverse_fisher_z_transform(z_low)
    RDM_mean = 1 - inverse_fisher_z_transform(z_mean)
    return RDM_mean, RDM_low, RDM_high




def fisher_z_transform(r):
    return np.arctanh(r)




def inverse_fisher_z_transform(z):
    return np.tanh(z)




def plot_RDM(RDM, filepath = None):
    plt.figure(figsize = (4, 3))
    sns.heatmap(RDM, cmap = "coolwarm", annot = True, fmt = '.2f', square = True, vmin = 0., vmax = 1.,
                cbar_kws = {'label': 'Dissimilarity'}, xticklabels = False, yticklabels = False, annot_kws = {'size': 14})
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Dissimilarity', fontsize = 18)
    cbar.tick_params(labelsize = 14)
    plt.tight_layout()
    plt.show()
    # plt.savefig(filepath, dpi = 300)
    # plt.close()




def compute_feedback_angle_error(corr, pre_angle_error, target_cond_idx):
    Max_diss = 1 - np.amax(corr)
    Max_sim_index = np.argmax(corr)            
    left_index, right_index = (Max_sim_index+1)%len(ANGLEs), (Max_sim_index-1)%len(ANGLEs)
    sim_neighbourL = corr[left_index]   
    sim_neighbourR = corr[right_index]
    angle_diff = 90
    if sim_neighbourL >= sim_neighbourR:           
        closeness = (Max_diss/(Max_diss+ 1 - sim_neighbourL)) 
        A = (ANGLEs[Max_sim_index]) + (closeness*angle_diff) 
    elif sim_neighbourR >= sim_neighbourL:
        closeness = (Max_diss/(Max_diss+ 1 - sim_neighbourR))
        A = (ANGLEs[Max_sim_index]) - (closeness*angle_diff)
    else:
        return pre_angle_error
    return min(360-abs(A-ANGLEs[target_cond_idx]), abs(A-ANGLEs[target_cond_idx]))




def compute_feedback_intensity(corr, NF_pattern, base_patterns, target_cond_idx):
    Max_sim_index = np.argmax(corr)
    if Max_sim_index != target_cond_idx: return 0.
    source_vec = NF_pattern
    target_vec = base_patterns[:, target_cond_idx]
    length_of_V2 = np.sqrt(np.dot(target_vec, target_vec))
    proj_of_V1_on_V2 = (np.dot(source_vec, target_vec)/np.dot(target_vec, target_vec))*target_vec 
    a1 = np.sqrt(sum(proj_of_V1_on_V2*proj_of_V1_on_V2))  
    intensity = a1/length_of_V2
    return min(int(intensity*5+1), 5)




def compute_feedback_correlations(corrs, block_mode):
    z_corrs = fisher_z_transform(corrs)
    z_corr = np.mean(z_corrs, axis = 0) if block_mode == "mean" else np.median(z_corrs, axis = 0)
    return inverse_fisher_z_transform(z_corr)




def compute_run_performance(sub_id, target_cond_idx, run_idx, NF_run_dir, base_patterns, block_mode):
    NF_correlations = np.nan_to_num(np.loadtxt(os.path.join(NF_run_dir, "all_rt_correlations.txt")))    # NF_TR * 4
    NF_patterns = np.nan_to_num(np.loadtxt(os.path.join(NF_run_dir, "all_rt_patterns.txt")))            # voxel_num * NF_TR        
    angular_distance = 180.
    angular_distances, intensities = [], []
    cond0_correlations, cond1_correlations, cond2_correlations, cond3_correlations = [], [], [], []
    run_df_time = pd.DataFrame({"sub_id": [], "condition": [], "run": [], "time": [], "angular_distance": [], "intensity": [], 
                                "cond0_correlation": [], "cond1_correlation": [], "cond2_correlation": [], "cond3_correlation": []})
    run_df_block  = pd.DataFrame({"sub_id": [], "condition": [], "run": [], "block": [], "angular_distance": [], "intensity": [],
                                  "cond0_correlation": [], "cond1_correlation": [], "cond2_correlation": [], "cond3_correlation": []})
    for TR_idx, corr in enumerate(NF_correlations):
        angular_distance = compute_feedback_angle_error(corr, angular_distance, target_cond_idx)
        intensity = compute_feedback_intensity(corr, NF_patterns[:, TR_idx], base_patterns, target_cond_idx)
        new_row_time = {"sub_id": sub_id, "condition": CONDITIONs[target_cond_idx], "run": f"Run {run_idx+1}", "time": TR_idx+1, 
                        "angular_distance": angular_distance, "intensity": intensity, 
                        "cond0_correlation": NF_correlations[TR_idx, 0], "cond1_correlation": NF_correlations[TR_idx, 1],
                        "cond2_correlation": NF_correlations[TR_idx, 2], "cond3_correlation": NF_correlations[TR_idx, 3]}
        run_df_time = pd.concat([run_df_time, pd.DataFrame([new_row_time])], ignore_index = True)
        angular_distances.append(angular_distance)
        intensities.append(intensity)
        cond0_correlations.append(NF_correlations[TR_idx, 0])
        cond1_correlations.append(NF_correlations[TR_idx, 1])
        cond2_correlations.append(NF_correlations[TR_idx, 2])
        cond3_correlations.append(NF_correlations[TR_idx, 3])
        if (TR_idx+1) % 6 == 0:
            block_idx = (TR_idx+1) // 6
            new_row_block = {"sub_id": sub_id, "condition": CONDITIONs[target_cond_idx], "run": f"Run {run_idx+1}", "block": block_idx, 
                             "angular_distance": np.mean(np.array(angular_distances)) if block_mode == "mean" else np.median(np.array(angular_distances)), 
                             "intensity": np.mean(np.array(intensities)) if block_mode == "mean" else np.median(np.array(intensities)), 
                             "cond0_correlation": compute_feedback_correlations(np.array(cond0_correlations), block_mode = block_mode),
                             "cond1_correlation": compute_feedback_correlations(np.array(cond1_correlations), block_mode = block_mode),
                             "cond2_correlation": compute_feedback_correlations(np.array(cond2_correlations), block_mode = block_mode),
                             "cond3_correlation": compute_feedback_correlations(np.array(cond3_correlations), block_mode = block_mode)}
            run_df_block = pd.concat([run_df_block, pd.DataFrame([new_row_block])], ignore_index = True)
            angular_distances, intensities, cond0_correlations, cond1_correlations, cond2_correlations, cond3_correlations = [], [], [], [], [], []
    return run_df_time, run_df_block




def compute_mean_correlations(corrs, axis = 0):
    z_corrs = fisher_z_transform(corrs)
    z_mean = np.mean(z_corrs, axis = axis)
    return inverse_fisher_z_transform(z_mean)




def correlation_high_ci(corrs):
    z_corrs = fisher_z_transform(corrs)
    z_mean = np.mean(z_corrs, axis = 0)
    z_std = np.std(z_corrs, axis = 0)
    z_high = z_mean + 1.96 * z_std / np.sqrt(z_corrs.shape[0])
    return inverse_fisher_z_transform(z_high)




def correlation_low_ci(corrs):
    z_corrs = fisher_z_transform(corrs)
    z_mean = np.mean(z_corrs, axis = 0)
    z_std = np.std(z_corrs, axis = 0)
    z_low = z_mean - 1.96 * z_std / np.sqrt(z_corrs.shape[0])
    return inverse_fisher_z_transform(z_low)




def normal_low_ci(corrs):
    return np.mean(corrs) - 1.96 * np.std(corrs) / np.sqrt(len(corrs))




def normal_high_ci(corrs):
    return np.mean(corrs) + 1.96 * np.std(corrs) / np.sqrt(len(corrs))




def plot_errorbar(init_final_df, target_metric, ax, x_ticks, offset = 0.2):
    for run_idx, tick in enumerate(x_ticks):
        init_vals = init_final_df[(init_final_df["run"] == f"Run {run_idx+1}") & (init_final_df["stage"] == "initial")][target_metric]
        final_vals = init_final_df[(init_final_df["run"] == f"Run {run_idx+1}") & (init_final_df["stage"] == "final")][target_metric]
        init_means, final_means = np.mean(init_vals), np.mean(final_vals)
        init_stds, final_stds = np.std(init_vals), np.std(final_vals)
        ax.scatter(tick-offset, init_means, color = "black", s = 10, zorder = 3)
        ax.scatter(tick+offset, final_means, color = "black", s = 10, zorder = 3)
        ax.errorbar(tick-offset, init_means, yerr = init_stds, color = "black", capsize = 2, capthick = 0.8, lw = 0.8, zorder = 2)
        ax.errorbar(tick+offset, final_means, yerr = final_stds, color = "black", capsize = 2, capthick = 0.8, lw = 0.8, zorder = 2)




def plot_marker(init_final_df, target_metric, ax, x_ticks, text_offset, offset = 0.2):
    for run_idx, tick in enumerate(x_ticks):
        init_vals = init_final_df[(init_final_df["run"] == f"Run {run_idx+1}") & (init_final_df["stage"] == "initial")][target_metric].to_numpy()
        final_vals = init_final_df[(init_final_df["run"] == f"Run {run_idx+1}") & (init_final_df["stage"] == "final")][target_metric].to_numpy()
        x1, x2 = tick-offset, tick+offset
        y = max(np.concatenate([init_vals, final_vals])) + text_offset
        line_offset = text_offset / 20
        differences = final_vals - init_vals
        shapiro_test = stats.shapiro(differences)
        if shapiro_test.pvalue > 0.05:
            stat, pval = stats.ttest_rel(final_vals, init_vals)
        else:
            stat, pval = stats.wilcoxon(final_vals, init_vals)
        pval = pval * 2             # Bonferroni correction (n = 2)
        print("t-test" if shapiro_test.pvalue > 0.05 else "wilcoxon", stat, pval)
        if pval < 0.05:
            if pval < 0.001: sig_marker = "***"
            elif pval < 0.01: sig_marker = "**"
            else: sig_marker = "*"
            ax.plot([x1, x1, x2, x2], [y, y+line_offset, y+line_offset, y], color = "black", linewidth = 1)
            ax.text((x1+x2)/2, y+line_offset, sig_marker, ha = "center", fontsize = 12, fontweight = 'bold')