# Hyper-Convolution Networks for Biomedical Image Segmentation
Code for our WACV 2022 paper: 

*Hyper-Convolution Networks for Biomedical Image Segmentation* (https://arxiv.org/abs/2105.10559)

and our journal extension published at Medical Image Analysis 

*Hyper-convolutions via implicit kernels for medical image analysis*
(https://www.sciencedirect.com/science/article/pii/S1361841523000579)

Convolutional Kernels are generated by a hyper-network instead of independtly learned

<img src="https://github.com/tym002/Hyper-Convolution/blob/main/architecture_nn.png" width="600">

The input to the hyper-network are the spatial coordinates of the kernels

## requirements: 

`tensorflow-gpu 1.15.0`

`python 3.6.13`

## Code:

To initiate training or testing, run:
`python main.py --mode train --config_path config.json`

`--mode train` for training, `--mode test` for testing

`--config_path` is the path to config json file that contains all model related config

`kernal.py` contains the input to the hyper-network, which is a two-channels coordinates grid (x and y)

`unet_vanilla.py` contains all the networks including the baseline UNet, non-local UNet and our method  

## Citation: 

If you find our code useful, please cite our work, thank you! 
```
@inproceedings{ma2022hyper,
  title={Hyper-convolution networks for biomedical image segmentation},
  author={Ma, Tianyu and Dalca, Adrian V and Sabuncu, Mert R},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1933--1942},
  year={2022}
}
```
```
@article{ma2023hyper,
  title={Hyper-convolutions via implicit kernels for medical image analysis},
  author={Ma, Tianyu and Wang, Alan Q and Dalca, Adrian V and Sabuncu, Mert R},
  journal={Medical Image Analysis},
  pages={102796},
  year={2023},
  publisher={Elsevier}
}
```
