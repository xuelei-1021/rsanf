#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns

from nilearn.image import new_img_like, load_img
from nilearn.plotting import plot_stat_map
from scipy import ndimage
from tqdm import tqdm


CONDITIONs = ["Joyfulrelaxation", "Sadness", "Enthusiasm", "Anger"]


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