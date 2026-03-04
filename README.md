# Knowledge-Aware Replay for Multi-Label Class-Incremental Learning

![](https://img.shields.io/badge/python-3.7-green)
![](https://img.shields.io/badge/torch-1.13.1-green)
![](https://img.shields.io/badge/cudatoolkit-11.7-green)

This repo provides a reference implementation of "Knowledge-Aware Replay for Multi-Label Class-Incremental Learning." 

# Abstract
Multi-Label Class-Incremental Learning (MLCIL) aims to adapt multi-label classifiers to evolving classes while mitigating catastrophic forgetting. The replay paradigm, which stores and revisits exemplars in a buffer, is a common enhancement in state-of-the-art MLCIL. However, its effectiveness is severely limited by the label-absence issue: instance annotations are restricted to their original task classes, ignoring potential co-occurring concepts. This causes two critical failures: (i) incompletely annotated exemplars convert correct detections into false negatives, and (ii) exemplar selection strategies inherited from single-label settings build biased buffers that overlook multi-label co-occurrence patterns. To address these challenges, we present Knowledge-Aware Replay (KAR), which reconciles the replay paradigm with MLCIL under the constraint of label absence. KAR features two core components. First, Bidirectional Knowledge Transfer (BKT) performs pseudo-labeling on both new data and replayed exemplars to recover missing labels, thereby correcting supervisory signals and reducing false negatives. Second, Knowledge-Informed Selection (KIS) constructs memory-efficient buffers by jointly optimizing representativeness and informativeness. This strategy captures evolving class distributions and prioritizes semantic-bridge instances to increase label density per exemplar. Extensive evaluations demonstrate that KAR consistently outperforms existing MLCIL baselines and exhibits excellent versatility.

# Dataset
## MS-COCO \& PASCAL-VOC
These two datasets are publicly available. 
## BDDOIA
The BDDOIA dataset is publicly available.
## FSD50K
The FSD50K dataset is publicly available.

# Pre-trained Model
The pre-trained TResNetM, NLE-DM and VGG are both publicly available.

# Environmental Settings
All experiments are conducted on an NVIDIA GeForce RTX 4090 GPU with 128GB RAM and Intel i9-14900K CPU, using `Python 3.7`, `PyTorch 1.13.1`, and `CUDA 11.7`.

**Step 1**: Install Anaconda

**Step 2**: Create a virtual environment and install the required packages
```shell
# create a new environment
conda create -n KAR python=3.7

# activate environment
conda activate KAR

# install Pytorch
pip install torch torchvision torchaudio

# install other required packages
pip install -r requirements.txt
```

# Usage
**Step 1**: Prepare datasets and pre-trained models.  
Download datasets and pre-trained models in the newly created folder [**datasets**] and [**pretrained_models**].

**Step 2**: Launch Experiments.
```shell
python main.py
```


# Results
| Method | Buffer | MS-COCO {10-10} | | | | MS-COCO {40-10} | | | |
|--------|--------|-----------------|---|---|---|-----------------|---|---|---|
| | | Average | Last | | | Average | Last | | |
| | | mAP | cF1 | oF1 | mAP | mAP | cF1 | oF1 | mAP |
| AGCN | 0 | 72.4 | 53.9 | 56.6 | 61.4 | 73.9 | 58.7 | 59.9 | 69.1 |
| KRT | 0 | 74.6 | 55.6 | 56.5 | 65.9 | 77.8 | 64.4 | 63.4 | 74.0 |
| APPLE | 0 | 75.6 | 56.4 | 58.7 | 66.3 | 78.4 | 64.8 | 65.6 | 75.9 |
| CSC | 0 | 78.0 | 64.9 | _66.8_ | 72.8 | 78.0 | _65.7_ | 67.0 | 75.0 |
| HCP | 0 | 77.9 | 60.4 | 65.3 | 71.2 | _78.9_ | 64.9 | _68.6_ | 75.3 |
| RebLL | 0 | _78.3_ | _65.4_ | 66.0 | _73.5_ | 78.7 | 65.3 | 66.5 | _76.1_ |
| **KAR** | 0 | **80.6**±0.1 | **68.2**±0.2 | **68.9**±0.1 | **75.2**±0.2 | **80.9**±0.1 | **67.8**±0.1 | **71.3**±0.2 | **78.2**±0.2 |
|--------|--------|---------|-----|-----|-----|---------|-----|-----|-----|
| AGCN-R | 20/class | 73.2 | 59.5 | 60.3 | 66.0 | 75.2 | 64.1 | 65.2 | 71.7 |
| KRT-R | 20/class | 76.5 | 63.9 | 64.7 | 70.2 | 78.3 | 67.9 | 68.9 | 75.2 |
| APPLE-R | 20/class | 76.8 | 64.1 | 65.6 | 70.5 | 78.2 | 67.3 | 68.4 | 74.6 |
| CSC-R | 20/class | _79.6_ | 67.8 | 68.6 | _74.8_ | 78.7 | 68.2 | 69.4 | 76.7 |
| HCP-R | 20/class | _79.6_ | _70.4_ | _73.0_ | 74.6 | _79.6_ | _71.9_ | _74.5_ | _77.2_ |
| RebLL | 20/class | 78.9 | 67.3 | 68.3 | 74.2 | 78.6 | 67.4 | 68.6 | 76.3 |
| **KAR** | 20/class | **81.2**±0.1 | **72.5**±0.2 | **75.6**±0.3 | **76.8**±0.2 | **81.4**±0.2 | **73.8**±0.1 | **77.6**±0.2 | **78.9**±0.1 |
|--------|--------|---------|-----|-----|-----|---------|-----|-----|-----|
| PRS | 1,000 | 48.8 | 8.5 | 14.7 | 27.9 | 50.8 | 9.3 | 15.1 | 33.2 |
| OCDM | 1,000 | 49.5 | 8.6 | 14.9 | 28.5 | 51.3 | 9.5 | 15.5 | 34.0 |
| AGCN-R | 1,000 | 73.0 | 59.4 | 65.6 | 59.0 | 75.0 | 63.1 | 64.8 | 71.1 |
| KRT-R | 1,000 | 75.7 | 61.6 | 63.6 | 69.3 | 78.3 | 67.5 | 68.5 | 75.1 |
| APPLE-R | 1,000 | 76.2 | 62.4 | 63.1 | 70.6 | 77.6 | 66.3 | 67.8 | 74.3 |
| CSC-R | 1,000 | 79.3 | 67.5 | 68.5 | 73.9 | 78.5 | 67.8 | 69.7 | 76.0 |
| HCP-R | 1,000 | _79.5_ | _70.2_ | _72.8_ | _75.4_ | _79.5_ | _71.8_ | _74.4_ | _76.7_ |
| RebLL | 1,000 | 78.5 | 67.4 | 68.1 | 74.0 | 78.3 | 67.2 | 68.6 | 76.6 |
| **KAR** | 1,000 | **81.3**±0.2 | **72.9**±0.1 | **74.4**±0.1 | **78.7**±0.3 | **81.5**±0.1 | **73.6**±0.2 | **76.1**±0.2 | **78.9**±0.2 |