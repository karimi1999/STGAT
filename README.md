# ST-GAT: Spatial-Temporal Graph-Augmented Transformer for Cellular Network Traffic Forecasting

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of **ST-GAT**, a novel hybrid architecture for city-scale cellular network traffic forecasting. 

ST-GAT successfully mitigates spatial over-smoothing by dynamically fusing a Multi-Head Spatial Self-Attention module (for capturing global, functional dependencies) with a Graph Convolutional Network (GCN) branch driven by Chebyshev polynomials (for capturing localized topological priors). This fusion is governed by our proposed **Adaptive Spatio-Topological Gating Mechanism**.

---

## 🏛️ Model Architecture

The ST-GAT architecture operates on dual spatial perspectives. While the Transformer branch learns dynamic, all-to-all spatial correlations, the GCN branch explicitly restricts the receptive field to a $K$-hop physical neighborhood. The Adaptive Gating mechanism then dynamically balances these two representations for each spatial node at every time step.

---

## 🏆 Key Results (Internet Traffic)

Our model achieves state-of-the-art performance on the highly volatile Milan Internet traffic dataset, significantly outperforming existing baselines.

| Method | MAE | RMSE | R² |
| :--- | :---: | :---: | :---: |
| ST-DenseNet | 125.49 | 209.56 | 0.9260 |
| MVSTGN | 88.37 | 163.33 | 0.9540 |
| GLSTTN (Baseline) | 87.09 | 147.90 | 0.9629 |
| **ST-GAT (Fixed Fusion)** | 81.68 | 138.55 | 0.9675 |
| **ST-GAT (Adaptive Gating)** | **77.73** | **136.89** | **0.9683** |

---

## ⚠️ Important Note for Google Colab Users
To ensure the code runs efficiently and to prevent `AssertionError: Torch not compiled with CUDA enabled`, **you must enable the GPU accelerator** before running the scripts.

1. Go to `Runtime` > `Change runtime type` in the Colab menu.
2. Select **T4 GPU** (or any available GPU) under *Hardware accelerator*.
3. Click **Save**.

---

## 🚀 Quick Start (Reproducing Pre-trained Results)

You can easily clone the repository, download the dataset, and evaluate the pre-trained ST-GAT model on Internet traffic directly in Google Colab using the following commands. The `gdown` command will automatically download the required dataset into the correct folder.

```bash
# 1. Clone the repository
!git clone https://github.com/karimi1999/STGAT.git

# 2. Navigate to the project directory
%cd /content/STGAT

# 3. Download the dataset (data_git_version.h5)
!gdown "1btxc4CVvvlhLyJV-m6-DbgIySZGi5UND" -O dataset/data_git_version.h5

# 4. Evaluate the model using pre-trained weights
!python train_colab.py -traffic='internet' -graphconv=1 -no-train
```

## 🧠 Training the Model from Scratch
If you wish to train the ST-GAT architecture from scratch on the Internet traffic dataset, simply run the training script without the `-no-train` flag:

```bash
!python train_colab.py -traffic='internet' -graphconv=1
```

## 📊 Visualizing Results and Error Analysis
Once the evaluation or training is complete, the predictions are saved in `results_data/STGAT.h5`. You can use the following scripts to generate publication-quality visual analyses.

### 1. Time-Series Prediction and Error (Target Cell)
This script plots the comparison between the ground truth and the ST-GAT prediction for a specific high-traffic cell s(50, 58), alongside its absolute error bar plot.

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

file_path = 'results_data/STGAT.h5'
with h5py.File(file_path, 'r') as f:
    pred = f['pred'][:]
    truth = f['truth'][:]

# Extract data for target cell: row 10, col 18 (representing spatial grid 50, 58)
true_cell = truth[:, 0, 10, 18]
pred_cell = pred[:, 0, 10, 18]
error_cell = true_cell - pred_cell
time_steps = np.arange(len(true_cell))

# Plot 1: True vs Predicted Line Plot
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(time_steps, true_cell, label='Ground Truth', color='black', linewidth=1.5)
ax1.plot(time_steps, pred_cell, label='Prediction', color='red', linewidth=1.5)

ax1.set_title('True vs Predicted Internet Traffic for cell s(50, 58)', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Time (Hours)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Internet Traffic Volume', fontsize=12, fontweight='bold')
ax1.set_xlim([0, len(time_steps)])
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()
fig1.savefig('traffic_prediction_line.svg', format='svg', bbox_inches='tight')

# Plot 2: Prediction Error Bar Plot
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(time_steps, error_cell, color='blue', width=0.8)

ax2.set_title('Differences Between True and Predicted Values for cell s(50, 58)', fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('Time (Hours)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Prediction Error (Ground Truth - Prediction)', fontsize=12, fontweight='bold')
ax2.set_xlim([0, len(time_steps)])
ax2.grid(True, linestyle='--', alpha=0.7)

ax2.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

plt.tight_layout()
fig2.savefig('traffic_error_bar.svg', format='svg', bbox_inches='tight')

print("Both Time-Series plots successfully generated and saved as SVG!")
```

### 2. Spatial Error Distribution (Heatmap)
This script generates a spatial heatmap of the Mean Absolute Error (MAE) across the 20x20 cropped spatial grid, demonstrating the spatial consistency of the model.

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'results_data/STGAT.h5'
with h5py.File(file_path, 'r') as f:
    pred = f['pred'][:]
    truth = f['truth'][:]

# Calculate Mean Absolute Error across temporal dimension
absolute_error = np.abs(truth - pred)
spatial_mae = np.mean(absolute_error, axis=(0, 1))

plt.figure(figsize=(9, 7))
sns.set_theme(style="white")

percentile_95 = np.percentile(spatial_mae, 95)

# Generate custom labels from 40 to 59 for the cropped 20x20 grid
grid_labels = list(range(40, 60))

# Apply custom labels to X and Y axes
ax = sns.heatmap(spatial_mae, cmap="YlOrRd", vmin=0, vmax=percentile_95,
                 cbar_kws={'label': 'Mean Absolute Error (MAE)'},
                 linewidths=0.5, linecolor='lightgray',
                 xticklabels=grid_labels, yticklabels=grid_labels)

# Keep labels horizontal
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.title('Spatial Distribution of Prediction Error (Heatmap)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Spatial Grid (Longitude)', fontsize=12)
plt.ylabel('Spatial Grid (Latitude)', fontsize=12)

plt.savefig('error_spatial_heatmap.svg', format='svg', bbox_inches='tight')

print("Heatmap successfully generated and saved as SVG!")
```

## 📝 Citation
If you find this repository or our proposed ST-GAT architecture useful in your research, please consider citing our paper:
> **ST-GAT: A Spatial-Temporal Graph-Augmented Transformer for Cellular Network Traffic Forecasting** > *(Full citation details will be provided upon publication)*
