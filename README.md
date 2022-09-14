# MSRN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Multi-scale Residual Network for Image Super-Resolution](https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf)
.

## Table of contents

- [MSRN-PyTorch](#msrn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Multi-scale Residual Network for Image Super-Resolution](#multi-scale-residual-network-for-image-super-resolution)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 31: `model_arch_name` change to `msrn_x2`.
- line 33: `in_channels` change to `3`.
- line 35: `out_channels` change to `3`.
- line 37: `upscale_factor` change to `3`.
- line 39: `mode` change to `test`.
- line 72: `model_weights_path` change to `./results/pretrained_models/MSRN_x2-DIV2K-e19a5cef.pth.tar`.
-

```bash
python3 test.py
```

### Train model

- line 31: `model_arch_name` change to `msrn_x2`.
- line 33: `in_channels` change to `3`.
- line 35: `out_channels` change to `3`.
- line 37: `upscale_factor` change to `3`.
- line 41: `mode` change to `train`.
- line 55: `pretrained_model_weights_path` change to `./results/pretrained_models/MSRN_x2-DIV2K-e19a5cef.pth.tar`.

```bash
python3 train_lsrresnet.py
```

### Resume train model

- line 31: `model_arch_name` change to `msrn_x2`.
- line 33: `in_channels` change to `3`.
- line 35: `out_channels` change to `3`.
- line 37: `upscale_factor` change to `3`.
- line 41: `mode` change to `train`.
- line 58: `resume` change to `samples/msrn_x2/epoch_xxx.pth.tar`.

```bash
python3 train_lsrresnet.py
```

## Result

Source of original paper
results: [https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       | 
|:-------:|:-----:|:----------------:|
|  Set5   |   2   | 38.08(**37.98**) |
|  Set5   |   3   | 34.38(**34.35**) |
|  Set5   |   4   | 32.07(**32.13**) |
|  Set5   |   8   | 26.59(**26.74**) |

```bash
# Download `MSRN_x2-DIV2K-e19a5cef.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input:

<span align="center"><img width="240" height="360" src="figure/comic_lr.png"/></span>

Output:

<span align="center"><img width="240" height="360" src="figure/comic_sr.png"/></span>

```text
Build `msrn_x2` model successfully.
Load `msrn_x2` model weights `./results/pretrained_models/MSRN_x2-DIV2K-e19a5cef.pth.tar` successfully.
SR image save to `./figure/comic_sr.png`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### Multi-scale Residual Network for Image Super-Resolution

_Juncheng Li, Faming Fang, Kangfu Mei, Guixu Zhang_ <br>

**Abstract** <br>
Recent studies have shown that deep neural networks can significantly improve the quality of single-image
super-resolution. Current researches tend to use deeper convolutional neural networks to enhance performance. However,
blindly increasing the depth of the network cannot ameliorate the network effectively. Worse still, with the depth of
the network increases, more problems occurred in the training process and more training tricks are needed. In this
paper, we propose a novel multi-scale residual network (MSRN) to fully exploit the image features, which outperform most
of the state-of-the-art methods. Based on the residual block, we introduce convolution kernels of different sizes to
adaptively detect the image features in different scales. Meanwhile, we let these features interact with each other to
get the most efficacious image information, we call this structure Multi-scale Residual Block (MSRB). Furthermore, the
outputs of each MSRB are used as the hierarchical features for global feature fusion. Finally, all these features are
sent to the reconstruction module for recovering the high-quality image.

[[Paper]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf)

```bibtex
@inproceedings{li2018multi,
  title={Multi-scale residual network for image super-resolution},
  author={Li, Juncheng and Fang, Faming and Mei, Kangfu and Zhang, Guixu},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={517--532},
  year={2018}
}
```
