# SwinJGP
Pytorch Implementation of "Learning-based Joint Geometric-Probabilistic Shaping for Digital Semantic Communication"
## Installation
Python 3.9<br>
CUDA 12.4
## Requirements
matplotlib==3.5.1
numpy==1.21.5
pandas==2.0.3
scikit-image==0.24.0
scipy==1.13.1
timm==1.0.15
torch==1.12.0+cu113
torchvision==0.13.0+cu113
## Training & Evaluation
For training, run the following command (as an example):
``` 
python main.py --mode 'train' --mod_method '64qam' --Training_strategy Proposed --channel awgn
```

For evaluation, run the following command (as an example):
``` 
python main.py --mode 'test' --mod_method '64qam' --Training_strategy Proposed --channel awgn --load_checkpoint 1
```
