# C2FTrans
This repo is the official implementation for:\
[C2FTrans: Coarse-to-Fine Transformers for Medical Image Segmentation.](https://arxiv.org/pdf/2206.14409.pdf)\
(The details of our C2FTrans can be found at the models directory in this repo or in the paper.)

## Requirements
* python 3.6
* pytorch 1.8.0
* torchvision 0.9.0
* more details please see the requirements.txt

## Datasets
* The ACDC dataset could be acquired from [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
* The ISIC 2018 dataset could be acquired from [here](https://challenge.isic-archive.com/data/)

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
