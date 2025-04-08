# ECCNAS: Efficient Crowd Counting Neural Architecture Search

This repository contains the official implementation of **ECCNAS**, a framework for efficient neural architecture search (NAS) tailored for crowd counting tasks. ECCNAS enables the discovery of high-performance, resource-efficient crowd counting models suitable for deployment on devices with varying computational budgets.



## Repository Structure

- **experiments/**: Contains various experiment setups and model variants:
  - `Branch_nolat/`: NAS without latency constraints
  - `FasterSegBranch/`: Faster segmentation branch
  - `FB_dense/`, `FB_dense12_pretrain/`: Dense feature branches
  - `FB_ghost/`: Ghost module variants
  - `FB_MCNN/`: Multi-column CNN variants
  - `FB_resnet/`, `FB_resnet34/`, `FB_res101SFCN/`: ResNet-based variants
  - `FB_vgg_*`: VGG-based variants with different kernel sizes
  - `PL-ImageNet/`: Pretraining on ImageNet
  - `FBNetBranch/`: Main ECCNAS search and training codebase
- **FBNetBranch/**:
  - `architecture_main_file.py`: Main entry for architecture search
  - `supernet_main_file.py`: Supernet training
  - `architecture_functions/`, `supernet_functions/`, `fbnet_building_blocks/`, `general_functions/`: Core modules for search, training, and model building


## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch (recommended >=1.4)
- CUDA-compatible GPU
- Other dependencies: numpy, torchvision, etc.



### Dataset Preparation

Prepare crowd counting datasets such as:

- ShanghaiTech
- UCF-QNRF
- NWPU-Crowd


### Training & Search

- **Pretrain backbone or supernet:**

```bash
python experiments/FBNetBranch/supernet_main_file.py --config your_config.yaml
```

- **Run architecture search:**

```bash
python experiments/FBNetBranch/architecture_main_file.py --config your_search_config.yaml
```

- **Evaluate searched models:**

```bash
python experiments/FBNetBranch/testload.py --model_path path_to_model.pth
```


## Results

ECCNAS achieves:

- Up to **96% reduction** in MACs with minimal accuracy loss
- Flexible trade-offs between speed and accuracy
- Competitive performance on multiple crowd counting benchmarks


---

## Citation

If you use this code or ideas from ECCNAS, please cite:

```bibtex
@article{10.1145/3465455,
  author = {Wang, Yabin and Ma, Zhiheng and Wei, Xing and Zheng, Shuai and Wang, Yaowei and Hong, Xiaopeng},
  title = {ECCNAS: Efficient Crowd Counting Neural Architecture Search},
  journal = {ACM Transactions on Multimedia Computing, Communications, and Applications},
  volume = {18},
  number = {1s},
  pages = {1--19},
  year = {2022},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3465455},
  url = {https://doi.org/10.1145/3465455}
}
```
