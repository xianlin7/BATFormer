# BATFormer
This repo is the official implementation for:\
[BATFormer: Towards Boundary-Aware Lightweight Transformer for Efficient Medical Image Segmentation.](https://ieeexplore.ieee.org/abstract/document/10101800)\
(The details of our BATFormer can be found at the models directory in this repo or in the paper.)

## Requirements
* python 3.6
* pytorch 1.8.0
* torchvision 0.9.0
* more details please see the requirements.txt

## Datasets
* The ACDC dataset could be acquired from [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/). The slice-level ACDC dataset preprocessed by us can be downloaded from [here](https://drive.google.com/file/d/18W_d8ho0Tl7TgPQXczOXZK5OUxtYkQdc/view?usp=share_link).
* The ISIC 2018 dataset could be acquired from [here](https://challenge.isic-archive.com/data/).\
(The dataset partitioning of ACDC follows [TransUNet](https://github.com/Beckschen/TransUNet) and the ISIC 2018 is divided randomly.)

## Training
Commands for training
```
python train.py
```
## Testing
Commands for testing
``` 
python test.py
```
## References
1. [vit-pytorch](https://github.com/lucidrains/vit-pytorch)
2. [MedT](https://github.com/jeya-maria-jose/Medical-Transformer)
