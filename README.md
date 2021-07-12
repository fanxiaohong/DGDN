# Deep Geometric Distillation Network for Compressive Sensing MRI

This repository contains the CS-MRI reconstruction pytorch codes for the following paper：  
Xiaohong Fan, Yin Yang, Jianping Zhang*, "Deep Geometric Distillation Network for Compressive Sensing MRI", 2021, [[pdf]]() 

These codes are adapted from ISTA-Net+ and FISTA-Net. These codes are built on PyTorch and tested on Ubuntu 18.04/20.04 (Python3.x, PyTorch>=0.4) with Intel Xeon CPU E5-2630 and Nvidia Tesla V100 GPU.

## 1.Test CS-MRI
1.1、Pre-trained models:
All pre-trained models for our paper are in './model_MRI'.

1.2、Prepare test data:
The original test set BrainImages_test is in './data/'.

1.3、Prepare code:
Open './Core_MRI-DGDN.py' and change the default run_mode to test in parser (parser.add_argument('--run_mode', type=str, default='test', help='train、test')).

3.4、Run the test script (Core_MRI-DGDN.py).

3.5、Check the results in './result/'.

## 2.Train CS-MRI
2.1、Prepare training data:
We use the same dataset and training data pairs as ISTA-Net+ for CS-MRI. Limited by the size of upload, we are unable to upload training data directly. Here we provide a link to download the training data (https://github.com/jianzhangcs/ISTA-Net-PyTorch) for you.

2.2、Prepare measurement matrix:
We fix the pseudo radial sampling masks the same as ISTA-Net+. The measurement matrixs are in './sampling_matrix/'.

2.3、Prepare code:
Open './Core_MRI-DGDN.py' and change the default run_mode to train in parser (parser.add_argument('--run_mode', type=str, default='train', help='train、test')).

2.4、Run the train script (Core_MRI-DGDN.py).

2.5、Check the results in './log_MRI/'.

## Contact
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at fanxiaohong@smail.xtu.edu.cn
