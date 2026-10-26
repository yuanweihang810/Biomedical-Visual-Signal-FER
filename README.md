# Robust Biomedical Visual Signal Processing for Facial Expression Recognition

This is the official PyTorch implementation for the paper: 
**"Robust Biomedical Visual Signal Processing for Facial Expression Recognition via Condition-Specific Contrast Stretching"** (Submitted to *Biomedical Signal Processing and Control*).

## 📊 Dataset
The self-built dataset (Training set: 30,240 images; Test set: 3,460 images) is permanently archived on Zenodo:
*   **DOI**: [https://doi.org/10.5281/zenodo.23252243](https://doi.org/10.5281/zenodo.23252243)
*   *Note: The dataset repository contains the training set and the independent multi-label test set. Model weights are available separately.*

## 🛠️ Environment Setup
We recommend using a conda environment with Python 3.10+ and PyTorch 2.0+.

```bash
pip install torch torchvision pandas numpy scikit-learn opencv-python tqdm thop
🚀 Training
To train the model, use train_rafdb.py. For example, to train the multi-branch model (K=4) on ResNet-18:

bash
python train_rafdb.py --backbone resnet18 --K 4 --epochs 80 --patience 30 --bs 64 --lr 5e-4
To train the single-branch baseline (K=1):

bash
python train_rafdb.py --backbone resnet18 --K 1 --no_transform --epochs 60 --patience 15 --bs 64 --lr 5e-4
📈 Evaluation
We provide various evaluation scripts:

Degraded Subsets Evaluation (Low-light & Low-resolution):

bash
python eval_degraded_subset.py --ckpt path/to/model.pth --backbone resnet18 --K 4
Efficiency Analysis (Params, FLOPs, Latency):

bash
python eval_inference_speed.py --backbone resnet18 --bs 1 --device cuda
📂 Code Structure
cale_plus.py: Core model definition (Condition-Specific Transform, Multi-Branch, Aggregation).

train_rafdb.py: Main training script for RAF-DB.

rafdb_dataset.py: Dataloader for the dataset.

film_condbn.py: Baseline implementations for FiLM and Conditional BatchNorm.

eval_*.py: Scripts for downstream evaluations (calibration, degraded subsets, TTA, etc.).

📧 Contact
For any questions, please contact: Weihang Yuan (ywh15958108437@bjfu.edu.cn)
