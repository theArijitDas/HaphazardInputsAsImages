# Haphazard Inputs as Images in Online Learning

Preprint Link: https://arxiv.org/abs/2504.02912

## Citation
Please consider citing the paper below if you are using the code provided in this repository.

```
@article{agarwal2025haphazard,
  title={Haphazard Inputs as Images in Online Learning},
  author={Agarwal, Rohit and Dessai, Aryan and Sekh, Arif Ahmed and Agarwal, Krishna and Horsch, Alexander and Prasad, Dilip K},
  journal={arXiv preprint arXiv:2504.02912},
  year={2025}
}
```

## Overview
The field of varying feature space in online learning settings, also known as haphazard inputs, is very prominent nowadays due to its applicability in various fields. However, the current solutions to haphazard inputs are model-dependent and cannot benefit from the existing advanced deep-learning methods, which necessitate inputs of fixed dimensions. Therefore, we propose to transform the varying feature space in an online learning setting to a fixed-dimension image representation on the fly. This simple yet novel approach is model-agnostic, allowing any vision-based models to be applicable for haphazard inputs, as demonstrated using ResNet and ViT. The image representation handles the inconsistent input data seamlessly, making our proposed approach scalable and robust. We show the efficacy of our method on four publicly available datasets.

This repository contains implementation of Haphazard Inputs as Images (HI2) model.

# Datasets
We use 4 different datasets for this project. The link of all the datasets can be found below. 

Download all the datasets and store them in Data folder in the root directory. Please download the datsets files form the given link below and place them inside their respective directories (see instructions for each dataset below...).

- ### magic04
    Data link: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope  
    Directory: `Data/magic04`  

- ### a8a
    Data link: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a8a  
    Directory: `Data/a8a`  

- ### SUSY
    Data link: https://archive.ics.uci.edu/dataset/279/susy  
    Directory: `Data/SUSY`  


- ### HIGGS
    Data link: https://archive.ics.uci.edu/dataset/280/higgs  
    Directory: `Data/higgs` 

## Dataset Preparation
### Variable P
We varied the availability of each feature independently by a uniform distribution of probability $p$, i.e., each auxilairy feature is available for $100p\%$. For more information about this, see the paper.

## Files
To run the models, see `Code/main.py`. 

## Dependencies
```
pip install -r Code/requirements.txt
```

## Running the code

```
python Code/main.py
```
