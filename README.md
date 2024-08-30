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

The look-back window (L) is set to 36 for ILI and 96 for all other datasets. The prediction window (T) varies: PEMS (T ∈ {12, 24, 48, 96}), ILI (T ∈ {24, 36, 48, 60}), and all others (T ∈ {96, 192, 336, 720}). *Avg* represents the average results across these four prediction windows. Bold red values indicate the best performance, while underlined blue values represent the second-best.

| Models     | Metric | MixMamba (Ours)          | iTransformer (2023)     | RLinear (2023) |          | PatchTST (2023) |          | Crossformer (2023) |          | TiDE (2023) |          | TimesNet (2023) |          | DLinear (2023) |          | SCINet (2022) |          | FEDformer (2022) |          | Stationary (2022) |          | Autoformer (2021) |          |
|------------|--------|-----------------|----------|---------------------|----------|----------------|----------|-----------------|----------|--------------------|----------|--------------|----------|-----------------|----------|-----------------|----------|----------------|----------|-------------------|----------|-------------------|----------|-------------------|----------|
|            |        | MSE             | MAE      | MSE                 | MAE      | MSE            | MAE      | MSE             | MAE      | MSE                | MAE      | MSE          | MAE      | MSE             | MAE      | MSE             | MAE      | MSE            | MAE      | MSE               | MAE      | MSE               | MAE      | MSE               | MAE      |
| **ETTm1**  | 96     | **0.318**       | **0.350**| 0.334               | 0.368    | 0.355          | 0.376    | _0.329_         | _0.367_  | 0.404              | 0.426    | 0.364        | 0.387    | 0.338           | 0.375    | 0.345           | 0.372    | 0.418          | 0.438    | 0.379             | 0.419    | 0.386             | 0.398    | 0.505             | 0.475    |
|            | 192    | **0.363**       | **0.372**| 0.377               | 0.391    | 0.391          | 0.392    | _0.367_         | _0.385_  | 0.450              | 0.451    | 0.398        | 0.404    | 0.374           | 0.387    | 0.380           | 0.389    | 0.439          | 0.450    | 0.426             | 0.441    | 0.459             | 0.444    | 0.553             | 0.496    |
|            | 336    | **0.391**       | **0.393**| 0.426               | 0.420    | 0.424          | 0.415    | _0.399_         | _0.410_  | 0.532              | 0.515    | 0.428        | 0.425    | 0.410           | 0.411    | 0.413           | 0.413    | 0.490          | 0.485    | 0.445             | 0.459    | 0.495             | 0.464    | 0.621             | 0.537    |
|            | 720    | **0.450**       | **0.427**| 0.491               | 0.459    | 0.487          | 0.450    | _0.454_         | _0.439_  | 0.666              | 0.589    | 0.487        | 0.461    | 0.478           | 0.450    | 0.474           | 0.453    | 0.595          | 0.550    | 0.543             | 0.490    | 0.585             | 0.516    | 0.671             | 0.561    |
|            | Avg    | **0.381**       | **0.386**| 0.407               | 0.410    | 0.414          | 0.407    | _0.387_         | _0.400_  | 0.513              | 0.496    | 0.419        | 0.419    | 0.400           | 0.406    | 0.403           | 0.407    | 0.485          | 0.481    | 0.448             | 0.452    | 0.481             | 0.456    | 0.588             | 0.517    |
| **ETTm2**  | 96     | _0.176_         | **0.254**| 0.180               | 0.264    | 0.182          | 0.265    | **0.175**       | _0.259_  | 0.287              | 0.366    | 0.207        | 0.305    | 0.187           | 0.267    | 0.193           | 0.292    | 0.286          | 0.377    | 0.203             | 0.287    | 0.192             | 0.274    | 0.255             | 0.339    |
|            | 192    | **0.241**       | **0.297**| 0.250               | 0.309    | 0.246          | 0.304    | _0.241_         | _0.302_  | 0.414              | 0.492    | 0.290        | 0.364    | 0.249           | 0.309    | 0.284           | 0.362    | 0.399          | 0.445    | 0.269             | 0.328    | 0.280             | 0.339    | 0.281             | 0.340    |
|            | 336    | **0.301**       | **0.337**| 0.311               | 0.348    | 0.307          | _0.342_  | _0.305_         | 0.343    | 0.597              | 0.542    | 0.377        | 0.422    | 0.321           | 0.351    | 0.369           | 0.427    | 0.637          | 0.591    | 0.325             | 0.366    | 0.334             | 0.361    | 0.339             | 0.372    |
|            | 720    | **0.400**       | **0.394**| 0.412               | 0.407    | 0.407          | _0.398_  | _0.402_         | 0.400    | 1.730              | 1.042    | 0.558        | 0.524    | 0.408           | 0.403    | 0.554           | 0.522    | 0.960          | 0.735    | 0.421             | 0.415    | 0.417             | 0.413    | 0.433             | 0.432    |
|            | Avg    | **0.280**       | **0.321**| 0.288               | 0.332    | 0.286          | 0.327    | _0.281_         | _0.326_  | 0.757              | 0.610    | 0.358        | 0.404    | 0.291           | 0.333    | 0.350           | 0.401    | 0.571          | 0.537    | 0.305             | 0.349    | 0.306             | 0.347    | 0.327             | 0.371    |
| **ETTh1**  | 96     | **0.374**       | **0.389**| 0.386               | 0.405    | 0.386          | _0.395_  | 0.414           | 0.419    | 0.423              | 0.448    | 0.479        | 0.464    | 0.384           | 0.402    | 0.386           | 0.400    | 0.654          | 0.599    | _0.376_           | 0.419    | 0.513             | 0.491    | 0.449             | 0.459    |
|            | 192    | **0.420**       | **0.417**| 0.441               | 0.436    | 0.437          | _0.424_  | 0.460           | 0.445    | 0.471              | 0.474    | 0.525        | 0.492    | _0.436_         | 0.429    | 0.437           | 0.432    | 0.719          | 0.631    | **0.420**         | 0.448    | 0.534             | 0.504    | 0.500             | 0.482    |
|            | 336    | _0.463_         | **0.439**| 0.487               | 0.458    | 0.479          | _0.446_  | 0.501           | 0.466    | 0.570              | 0.546    | 0.565        | 0.515    | 0.491           | 0.469    | 0.481           | 0.459    | 0.778          | 0.659    | **0.459**         | 0.465    | 0.588             | 0.535    | 0.521             | 0.496    |
|            | 720    | **0.455**       | **0.454**| 0.503               | 0.491    | _0.481_        | _0.470_  | 0.500           | 0.488    | 0.653              | 0.621    | 0.594        | 0.558    | 0.521           | 0.500    | 0.519           | 0.516    | 0.836          | 0.699    | 0.506             | 0.507    | 0.643             | 0.616    | 0.514             | 0.512    |
|            | Avg    | **0.429**       | **0.428**| 0.454               | 0.447    | 0.446          | _0.434_  | 0.469           | 0.454    | 0.529              | 0.522    | 0.541        | 0.507    | 0.458           | 0.450    | 0.456           | 0.452    | 0.747          | 0.647    | _0.440_           | 0.460    | 0.570             | 0.537    | 0.496             | 0.487    |
| **ETTh2**  | 96     | **0.283**       | **0.329**| 0.297               | 0.349    | _0.288_        | _0.338_  | 0.302           | 0.348    | 0.745              | 0.584    | 0.400        | 0.440    | 0.340           | 0.374    | 0.333           | 0.387    | 0.707          | 0.621    | 0.358             | 0.397    | 0.476             | 0.458    | 0.346             | 0.388    |
|            | 192    | **0.363**       | **0.380**| 0.380               | 0.400    | _0.374_        | _0.390_  | 0.388           | 0.400    | 0.877              | 0.656    | 0.528        | 0.509    | 0.402           | 0.414    | 0.477           | 0.476    | 0.860          | 0.689    | 0.429             | 0.439    | 0.512             | 0.493    | 0.456             | 0.452    |
|            | 336    | **0.406**       | **0.416**| 0.428               | 0.432    | _0.415_        | _0.426_  | 0.426           | 0.433    | 1.043              | 0.731    | 0.643        | 0.571    | 0.452           | 0.452    | 0.594           | 0.541    | 1.000          | 0.744    | 0.496             | 0.487    | 0.552             | 0.551    | 0.482             | 0.486    |
|            | 720    | **0.415**       | **0.433**| 0.427               | 0.445    | _0.420_        | _0.440_  | 0.431           | 0.446    | 1.104              | 0.763    | 0.874        | 0.679    | 0.462           | 0.468    | 0.831           | 0.657    | 1.249          | 0.838    | 0.463             | 0.474    | 0.562             | 0.560    | 0.515             | 0.511    |
|            | Avg    | **0.367**       | **0.390**| 0.383               | 0.407    | _0.374_        | _0.398_  | 0.387           | 0.407    | 0.942              | 0.684    | 0.611        | 0.550    | 0.414           | 0.427    | 0.559           | 0.515    | 0.954          | 0.723    | 0.437             | 0.449    | 0.526             | 0.516    | 0.450             | 0.459    |
| **PEMS03** | 12     | 0.072           | 0.175    | _0.071_             | _0.174_  | 0.126          | 0.236    | 0.099           | 0.216    | 0.090              | 0.203    | 0.178        | 0.305    | 0.085           | 0.192    | 0.122           | 0.243    | **0.066**      | **0.172**| 0.126             | 0.251    | 0.081             | 0.188    | 0.272             | 0.385    |
|            | 24     | _0.091_         | _0.199_  | 0.093               | 0.201    | 0.246          | 0.334    | 0.142           | 0.259    | 0.121              | 0.240    | 0.257        | 0.371    | 0.118           | 0.223    | 0.201           | 0.317    | **0.085**      | **0.198**| 0.149             | 0.275    | 0.105             | 0.214    | 0.334             | 0.440    |
|            | 48     | **0.121**       | **0.233**| _0.125_             | _0.236_  | 0.551          | 0.529    | 0.211           | 0.319    | 0.202              | 0.317    | 0.379        | 0.463    | 0.155           | 0.260    | 0.333           | 0.425    | 0.127          | 0.238    | 0.227             | 0.348    | 0.154             | 0.257    | 1.032             | 0.782    |
|            | 96     | **0.162**       | **0.270**| _0.164_             | _0.275_  | 1.057          | 0.787    | 0.269           | 0.370    | 0.262              | 0.367    | 0.490        | 0.539    | 0.228           | 0.317    | 0.457           | 0.515    | 0.178          | 0.287    | 0.348             | 0.434    | 0.247             | 0.336    | 1.031             | 0.796    |
|            | Avg    | **0.112**       | _0.222_  | 0.113               | 0.221    | 0.495          | 0.472    | 0.180           | 0.291    | 0.169              | 0.281    | 0.326        | 0.419    | 0.147           | 0.248    | 0.278           | 0.375    | 0.114          | 0.224    | 0.213             | 0.327    | 0.147             | 0.249    | 0.667             | 0.601    |
| **PEMS08** | 12     | 0.090           | 0.186    | **0.079**           | **0.182**| 0.133          | 0.247    | 0.168           | 0.232    | 0.165              | 0.214    | 0.227        | 0.343    | 0.112           | 0.212    | 0.154           | 0.276    | _0.087_        | _0.184_  | 0.173             | 0.273    | 0.109             | 0.207    | 0.436             | 0.485    |
|            | 24     | 0.125           | 0.226    | **0.115**           | **0.219**| 0.249          | 0.343    | 0.224           | 0.281    | 0.215              | 0.260    | 0.318        | 0.409    | 0.141           | 0.238    | 0.248           | 0.353    | _0.122_        | _0.221_  | 0.210             | 0.301    | 0.140             | 0.236    | 0.467             | 0.502    |
|            | 48     | **0.184**       | _0.237_  | _0.186_             | **0.235**| 0.569          | 0.544    | 0.321           | 0.354    | 0.315              | 0.355    | 0.497        | 0.510    | 0.198           | 0.283    | 0.440           | 0.470    | 0.189          | 0.270    | 0.320             | 0.394    | 0.211             | 0.294    | 0.966             | 0.733    |
|            | 96     | **0.216**       | **0.262**| _0.221_             | _0.267_  | 1.166          | 0.814    | 0.408           | 0.417    | 0.377              | 0.397    | 0.721        | 0.592    | 0.320           | 0.351    | 0.674           | 0.565    | 0.236          | 0.300    | 0.442             | 0.465    | 0.345             | 0.367    | 1.385             | 0.915    |
|            | Avg    | _0.154_         | _0.228_  | **0.150**           | **0.226**| 0.529          | 0.487    | 0.280           | 0.321    | 0.268              | 0.307    | 0.441        | 0.464    | 0.193           | 0.271    | 0.379           | 0.416    | 0.158          | 0.244    | 0.286             | 0.358    | 0.201             | 0.276    | 0.814             | 0.659    |
| **Exchange**| 96    | **0.084**       | **0.200**| _0.086_             | 0.206    | 0.093          | 0.217    | _0.088_         | _0.205_  | 0.256              | 0.367    | 0.094        | 0.218    | 0.107           | 0.234    | 0.088           | 0.218    | 0.267          | 0.396    | 0.148             | 0.278    | 0.111             | 0.237    | 0.197             | 0.323    |
|            | 192    | **0.174**       | **0.295**| 0.177               | 0.299    | 0.184          | 0.307    | _0.176_         | _0.299_  | 0.470              | 0.509    | 0.184        | 0.307    | 0.226           | 0.344    | 0.176           | 0.315    | 0.351          | 0.459    | 0.271             | 0.315    | 0.219             | 0.335    | 0.300             | 0.369    |
|            | 336    | 0.333           | _0.415_  | 0.331               | 0.417    | 0.351          | 0.432    | **0.301**       | **0.397**| 1.268              | 0.883    | 0.349        | 0.431    | 0.367           | 0.448    | _0.313_         | 0.427    | 1.324          | 0.853    | 0.460             | 0.427    | 0.421             | 0.476    | 0.509             | 0.524    |
|            | 720    | **0.826**       | **0.682**| 0.847               | _0.691_  | 0.886          | 0.714    | 0.901           | 0.714    | 1.767              | 1.068    | 0.852        | 0.698    | 0.964           | 0.746    | _0.839_         | 0.695    | 1.058          | 0.797    | 1.195             | 0.695    | 1.092             | 0.769    | 1.447             | 0.941    |
|            | Avg    | _0.360_         | **0.401**| 0.360               | _0.403_  | 0.378          | 0.417    | _0.367_         | 0.404    | 0.940              | 0.707    | 0.370        | 0.413    | 0.416           | 0.443    | **0.354**       | 0.414    | 0.750          | 0.626    | 0.519             | 0.429    | 0.461             | 0.454    | 0.613             | 0.539    |
| **Weather**| 96     | 0.179           | **0.214**| 0.174               | **0.214**| 0.192          | 0.232    | 0.177           | _0.218_  | **0.158**          | 0.230    | 0.202        | 0.261    | _0.172_         | 0.220    | 0.196           | 0.255    | 0.221          | 0.306    | 0.217             | 0.296    | 0.173             | 0.223    | 0.266             | 0.336    |
|            | 192    | 0.226           | **0.254**| 0.221               | _0.254_  | 0.240          | 0.271    | 0.225           | 0.259    | **0.206**          | 0.277    | 0.242        | 0.298    | _0.219_         | 0.261    | 0.237           | 0.296    | 0.261          | 0.340    | 0.276             | 0.336    | 0.245             | 0.285    | 0.307             | 0.367    |
|            | 336    | 0.281           | **0.293**| _0.278_             | _0.296_  | 0.292          | 0.307    | 0.278           | 0.297    | **0.272**          | 0.335    | 0.287        | 0.335    | 0.280           | _0.306_  | 0.283           | 0.335    | 0.309          | 0.378    | 0.339             | 0.380    | 0.321             | 0.338    | 0.359             | 0.395    |
|            | 720    | 0.355           | **0.342**| 0.358               | _0.347_  | 0.364          | 0.353    | 0.354           | 0.348    | 0.398              | 0.418    | _0.351_      | 0.386    | 0.365           | 0.359    | **0.345**       | 0.381    | 0.377          | 0.427    | 0.403             | 0.428    | 0.414             | 0.410    | 0.419             | 0.428    |
|            | Avg    | 0.261           | **0.277**| **0.258**           | _0.278_  | 0.272          | 0.291    | _0.259_         | 0.281    | 0.259              | 0.315    | 0.271        | 0.320    | 0.259           | 0.287    | 0.265           | 0.317    | 0.292          | 0.363    | 0.309             | 0.360    | 0.288             | 0.314    | 0.338             | 0.382    |
| **ILI**    | 24     | **1.971**       | **0.838**| 2.472               | 0.994    | 5.742          | 1.772    | _2.290_         | _0.920_  | 3.906              | 1.332    | 5.452        | 1.732    | 2.317           | 0.934    | 2.398           | 1.040    | 3.687          | 1.420    | 3.228             | 1.260    | _2.294_           | _0.945_  | 3.483             | 1.287    |
|            | 36     | _1.875_         | **0.816**| 2.288               | 0.964    | 5.343          | 1.672    | 2.345           | 0.928    | 3.880              | 1.278    | 4.960        | 1.621    | **1.972**       | 0.920    | 2.646           | 1.088    | 3.941          | 1.582    | 2.679             | 1.080    | 1.825             | _0.848_  | 3.103             | 1.148    |
|            | 48     | **1.898**       | **0.829**| 2.227               | 0.951    | 4.722          | 1.563    | _2.213_         | _0.916_  | 3.896              | 1.273    | 4.561        | 1.533    | 2.238           | 1.982    | 2.614           | 1.086    | 3.193          | 1.202    | 2.622             | 1.078    | _2.010_           | _0.900_  | 2.669             | 1.085    |
|            | 60     | **1.893**       | **0.849**| 2.267               | 0.966    | 4.526          | 1.529    | _2.143_         | _0.904_  | 4.190              | 1.331    | 4.632        | 1.556    | _2.027_         | 0.928    | 2.804           | 1.146    | 3.187          | 1.198    | 2.857             | 1.157    | 2.178             | 0.963    | 2.770             | 1.125    |
|            | Avg    | **1.909**       | **0.858**| 2.302               | 0.968    | 5.083          | 1.621    | _2.247_         | _0.917_  | 3.968              | 1.303    | 4.894        | 1.610    | 2.139           | 0.931    | 2.616           | 1.090    | 3.502          | 1.350    | 2.847             | 1.144    | _2.077_           | 0.914    | 3.006             | 1.161    |
| **ECL**    | 96     | _0.152_         | _0.248_  | **0.148**           | **0.240**| 0.201          | 0.281    | 0.195           | 0.285    | 0.219              | 0.314    | 0.237        | 0.329    | 0.168           | 0.272    | 0.197           | 0.282    | 0.247          | 0.345    | 0.193             | 0.308    | _0.169_           | _0.273_  | 0.201             | 0.317    |
|            | 192    | _0.189_         | _0.276_  | **0.162**           | **0.253**| 0.201          | 0.283    | 0.199           | 0.289    | 0.231              | 0.322    | 0.236        | 0.330    | 0.184           | 0.289    | 0.196           | 0.285    | 0.257          | 0.355    | 0.201             | 0.315    | _0.182_           | _0.286_  | 0.222             | 0.334    |
|            | 336    | 0.204           | 0.301    | **0.178**           | **0.269**| 0.215          | 0.298    | 0.215           | 0.305    | 0.246              | 0.337    | 0.249        | 0.344    | 0.198           | _0.300_  | 0.209           | 0.301    | 0.269          | 0.369    | 0.214             | 0.329    | _0.200_           | _0.304_  | 0.231             | 0.338    |
|            | 720    | 0.226           | 0.322    | _0.225_             | **0.317**| 0.257          | 0.331    | 0.256           | 0.337    | 0.280              | 0.363    | 0.284        | 0.373    | **0.220**       | _0.320_  | 0.245           | 0.333    | 0.299          | 0.390    | 0.246             | 0.355    | _0.222_           | _0.321_  | 0.254             | 0.361    |
|            | Avg    | 0.193           | _0.287_  | **0.178**           | **0.270**| 0.219          | 0.298    | 0.216           | 0.304    | 0.244              | 0.334    | 0.251        | 0.344    | _0.192_         | 0.295    | 0.212           | 0.300    | 0.268          | 0.365    | 0.214             | 0.327    | 0.193             | _0.296_  | 0.227             | 0.338    |
| **Count**  | 1st Count | **29**        | **38**   | _8_                | _10_     | 0              | 0        | 2               | 1        | 3                  | 0        | 0            | 0        | 1               | 0        | 2               | 0        | 2              | 2        | 2                 | 0        | 1                 | 0        | 0                 | 0        |


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
