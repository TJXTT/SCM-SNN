# Spike Count Maximization for Neuromorphic Vision Recognition
The matlab code for the paper "Spike Count Maximization for Neuromorphic Vision Recognition".

## Usage
### Prerequisites

* Matlab 2018a

### Example for Running
1. Download the [example data](https://drive.google.com/drive/folders/1-AWTazb5P1Vt6p-fowaWdvHIAqhuLPEQ?usp=drive_link) and extract ```baseline_classifier.mat```, ```train_data.mat```, and ```test_data.mat``` into the ```example``` folder. The example data is the CIFAR10-DVS feature set. It was extracted from a PLIF-based Spiking ResNet-18 trained with MSE loss and [PiecewiseLeakyReLU](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/surrogate.py) surrogate gradient.
2. Start Matlab, and then run the ```run.m``` script.

## Citation
If our work is helpful to you, please kindly cite our paper as:
```
@inproceedings{tang2023scm,
  title={Spike Count Maximization for Neuromorphic Vision Recognition},
  author={Jianxiong Tang and Jian-Huang Lai and Xiaohua Xie and Lingxiao Yang},
  booktitle={IJCAI},
  year={2023}
}
```
