# Navigating emotional states via RSA-informed real-time fMRI semantic neurofeedback (rt-fMRI-sNF)

This repository contains code from our feasibility study on **Toward navigating emotional states using real-time representational similarity analysis fMRI neurofeedback**.  
The project investigates how multivariate brain patterns can be decoded in real time and used to guide participants in modulating emotional states through mental imagery.

---

## Repository Structure

- **scripts/** – scripts for univariate and multivariate analyses (GLM, RSA, ROI analyses, etc.)  
- **data/** – real-time and offline preprocessed fMRI data (not available here)  
- **results/** – statistical results and corresponding figures (linear mixed-effects models, univariate stats, logs, etc.) (not available here)  
---

## Requirements

- Python 3.6.12  
- [Nilearn](https://nilearn.github.io/)  
- [Statsmodels](https://www.statsmodels.org/)  
- [Scikit-learn](https://scikit-learn.org/)  
- [NumPy](https://numpy.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [Matplotlib](https://matplotlib.org/)  

(See [`requirements.txt`](./requirements.txt) for the full list with specific version numbers.)

---

## Usage

### 1. Preprocessing
Preprocessed fMRI data (e.g., from **fMRIPrep**) and real-time output are required as input.  

### 2. Running Analyses
- **Univariate analysis**: `1_univ_pattern.ipynb`  
- **Multivariate (RSA) analysis**: `2_rsa_pattern.ipynb`  
- **Group-level neurofeedback analysis**: `3_group_NF.ipynb`  
- **Individual-level neurofeedback analysis**: `4_individual_NF.ipynb`  

---

## Reproducibility

All key analyses from the manuscript can be reproduced with the scripts provided here.  
The dataset will be made available at a later stage (see manuscript statement).  

---

## Citation

If you use this code, please cite:  

**Toward navigating emotional states using real-time representational similarity analysis fMRI neurofeedback - a feasibility study**.

---

## Contact

For questions or issues, please open a GitHub issue or contact:  
**Xuelei Wang** – [xuwang@ukaachen.de]  

---
