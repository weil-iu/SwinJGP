# SwinJGP
Pytorch Implementation of "Learning-based Joint Geometric-Probabilistic Shaping for Digital Semantic Communication"
## Installation
Python 3.9<br>
CUDA 12.4
## Requirements

## Training & Evaluation
For training, run the following command (as an example):
``` 
python main.py --mode 'train' --mod_method '64qam' --Training_strategy Proposed --channel awgn
```

For evaluation, run the following command (as an example):
``` 
python main.py --mode 'test' --mod_method '64qam' --Training_strategy Proposed --channel awgn --load_checkpoint 1
```
