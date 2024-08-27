# MixMamba
This repository is the official implementation of the paper: [MixMamba: Time Series Modeling with Adaptive Expertise](https://www.sciencedirect.com/science/article/pii/S1566253524003671)

## Introduction
The heterogeneity and non-stationary characteristics of time series data continue to challenge single models’ ability to capture complex temporal dynamics, especially in long-term forecasting. Therefore, we propose MixMamba that:
+ Leverages the [Mamba](https://arxiv.org/abs/2312.00752) model as an expert within a mixture-of-experts ([MoE](https://arxiv.org/abs/2308.00951)). This framework decomposes modeling into a pool of specialized experts,
enabling the model to learn robust representations and capture the full spectrum of patterns present in time series data. 
+ A dynamic gating network is introduced to adaptively allocates each data segment to the most suitable expert based on its characteristics allows the model to adjust dynamically to temporal changes in the underlying data distribution.
+ To prevent bias towards a limited subset of experts, a load balancing loss function is incorporated. 

## Overall Architecture
Schematic architecture of MixMamba is shown below. The process begins with pre-processing the raw time series data through normalization and segmentation (left). These patches are
then embedded and augmented with positional information before being input into the mixture of Mamba (MoM) block (center). This block consists of multiple Mamba experts
coordinated via a gating network (right). Each Mamba module includes a series of projections, convolution, selective SSM, and a skip connection to learn temporal dependencies.
Finally, a linear prediction head is employed to generate final outputs

## Datasets
Well-preprocessed datasets can be downloaded from either [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). After downloading, place the data in the ```./dataset``` folder.

## Acknowledgement
We'd like to express our gratitude to the following GitHub repositories for their exceptional codebase:

https://github.com/thuml/Time-Series-Library

https://github.com/yuqinie98/PatchTST

https://github.com/thuml/iTransformer
