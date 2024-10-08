# MixMamba
This repository is the official implementation of the paper: [MixMamba: Time Series Modeling with Adaptive Expertise](https://www.sciencedirect.com/science/article/pii/S1566253524003671)

## Introduction
The heterogeneity and non-stationary characteristics of time series data continue to challenge single models’ ability to capture complex temporal dynamics, especially in long-term forecasting. Therefore, we propose **MixMamba** that:
+ Leverages the [Mamba](https://arxiv.org/abs/2312.00752) model as an expert within a _mixture-of-experts_ ([MoE](https://arxiv.org/abs/2308.00951)). This framework decomposes modeling into a pool of specialized experts,
enabling the model to learn robust representations and capture the full spectrum of patterns present in time series data. 
+ A dynamic gating network is introduced to adaptively allocates each data segment to the most suitable expert based on its characteristics allows the model to adjust dynamically to temporal changes in the underlying data distribution.
+ To prevent bias towards a limited subset of experts, a load balancing loss function is incorporated.

## Schematic Architecture
MixMamba is a time series forecasting model that utilizes a mixture-of-experts (MoM) approach. The model's architecture consists of four primary stages:
- **Pre-processing**: Raw time series data undergoes normalization and segmentation to create patches.
- **Embedding and Augmentation**: Patches are embedded and augmented with positional information to provide context.
- **MoM Block**: This central component consists of multiple Mamba experts coordinated by a gating network. Each Mamba expert employs a series of projections, convolutions, selective SSM, and a skip connection to learn temporal dependencies.
- **Prediction Head**: A linear prediction head is used to generate final outputs based on the learned representations.
  
![architecture](img/MixMamba_architecture.png)


## Algorithms
<p align="center">
  <img src="img/algo1.png" width="30%" />
  <img src="img/algo2.png" width="30%" />
  <img src="img/algo3.png" width="30%" />
</p>

## Visualization
<figure>
  <figcaption>
    <div align="center"><b>Long-term Forecasting with 𝐿 = 96 and 𝑇 = 192 on ETTh1.</b></div>
  </figcaption>
  <img src="img/ETTh1.png" alt="Long-term Forecasting with 𝐿 = 96 and 𝑇 = 192 on ETTh1." width="100%">
</figure>


<figure align="center">
    <figcaption>
    <div align="center"><b>Long-term Forecasting with 𝐿 = 96 and 𝑇 = 192 on Weather.</b></div>
  </figcaption>
  <img src="img/Weather.png" alt="Long-term Forecasting with 𝐿 = 96 and 𝑇 = 192 on Weather" width="100%">
</figure>

<figure>
    <figcaption>
    <div align="center"><b>Short-term Forecasting on M4 (Yearly).</b></div>
  </figcaption>
  <img src="img/M4.png" alt="Description of picture 3" width="100%">
</figure>

## Install
Please follow the guide here to prepare the environment on **Linux OS**.
1. Clone this repository
```bash
git clone https://github.com/KhaledAlkilane89/MixMamba.git
cd MixMamba
```
2. Create environment and install package:
```bash
conda create -n mixmamba python=3.10 -y
conda activate mixmamba
pip install -r requirements.txt
```
3. Datasets can be downloaded from either [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). After downloading, place the data in the ```./dataset``` folder.

## Usage
Train and evaluate the model using the scripts provided in the ```./scripts/``` directory. 
Please refer to the following example for reproducing the experimental results:
- **Long-term forecasting**:
```bash ./scripts/long_term_forecast/ETT_script/mixmamba_ETTh1.sh```
- **Short-term Forecasting**:
```bash ./scripts/short_term_forecast/mixmamba_M4.sh```
- **Classification**:
```bash ./scripts/classification/mixmamba.sh```

## Main Results
## Long-term forecasting performance on various datasets
<figure>
    <figcaption>
    <div align="center"><b>Multivariate Long-term Forecasting.</b></div>
  </figcaption>
  <img src="img/long_term_results.png" alt="long_term_results" width="100%">
</figure>


<figure align="center">
    <figcaption>
    <div align="center"><b>Multivariate Short-term Forecasting.</b></div>
  </figcaption>
  <img src="img/short_term_results.png" alt="short_term_results" width="100%">
</figure>

<figure>
  <figcaption>
    <div align="center"><b>Classification.</b></div>
  </figcaption>
  <img src="img/classification_results.png" alt="classification" width="100%">
</figure>

## Model Analysis
- Mixmamba performance under varied look-back window length $𝐿 ∈ {96, 192, 336, 720}$ on PEMS03 datasets ($𝑇 = 720$) (**Upper left**).
- Comparison of memory usage (_Up_) and computation time (_Down_) on ETTm2 dataset (Batch size is set to 32) (**Upper right**).
- Comparison of learned representations for different experts on ETTm1 dataset with $𝐿 = 96, 𝑇 = 720$  (**Down left**).
- Hyperparameters analysis on exchange and ILI datasets ($𝐿 = 96, 𝑇 = 720$). (**Down right**)
<p align="center">
<img src="img/Lookback_window.png" width="45%" />
<img src="img/Time_complexity.png" width="45%" />
</p>
<p align="center">
<img src="img/Representations.png" width="45%" />
<img src="img/Hyperparameters.png" width="45%" />
</p>

## Citation
If you use this code or data in your research, please cite:

```bibtex
@article{ALKILANE2024102589,
title = {MixMamba: Time series modeling with adaptive expertise},
author = {Khaled Alkilane and Yihang He and Der-Horng Lee},
journal = {Information Fusion},
volume = {112},
pages = {102589},
year = {2024},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2024.102589},
url = {https://www.sciencedirect.com/science/article/pii/S1566253524003671}
}
```
## Contact Information
For inquiries or to discuss potential code usage, please reach out to the following researchers:
- Khaled (khaledalkilane@outlook.com)
- Yihang (yihang.23@intl.zju.edu.cn)
  
## Acknowledgement
We'd like to express our gratitude to the following GitHub repositories for their exceptional codebase:
- https://github.com/lucidrains/mixture-of-experts
- https://github.com/lucidrains/st-moe-pytorch
- https://github.com/thuml/Time-Series-Library
- https://github.com/yuqinie98/PatchTST
- https://github.com/thuml/iTransformer
