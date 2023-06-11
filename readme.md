# Brain-wide SNR improvement of neuron connections using deep learning

## Overview:
This study aimed to improve the brain-wide low signal-to-noise ratio (SNR) of optical neuronal image at cellular level using deep-learning. As low SNR widely exists despite progresses from labelling, imaging and processing, operators had to give up ~19% reconstruction of low SNR and dense images in recent trials. This study developed an automatic, typical and feature balanced training set construction method for brain-scale neuronal image. Previous DL based methods were usually built for specific local images. The application of DL to massive-scale has encountered the problem of selecting training examples to represent and balance the high diverse and complex features of whole-brain. Currently, operators have to use large-scale examples or iterative experiments (including transfer training) to build the training set, which are usually subjective, unrepeatability and expensive in training 3D examples. I have simplified the construction using specific heuristic knowledge, and explored key, quantitative and distinguishable indicators to characterize the diversity and hardness of targeted neuronal regions. Thus, a typical training set is built based on specific rules with less examples and no experience dependency. 

## System Requirements
### Hardware Requirements:
The deep learning algorithm requires enough RAM and GPU to support the calculation. For optimal performace, we recommenda computer with the following specs:
RAM: 16+GB
CPU: Intel i7 or better
GPU:  1080Ti or better

### Software Requirements:
The package development version is tested on Linux operating systems. The developmental version of the package has been tested on the following systems:
Linux: Ubuntu 16.04.1

### Environment Requirements:
Nvidia GPU corresponding driver
CUDA: cuda 9.0
cudnn: cudnn 7
Python: 3.6
pytorch:0.4.1 
visdom:0.1.8.5
Numpy: 1.14.5
tifffile: 0.15.1
Scikit-image:0.13.1

### Functions:
For interactive demos of the functions, please give the file paths that include the training and testing images. You can also adjust some paramters for better training or testing in your own computer. The python file config.py is used for configuration of the packages.  Paths and training or testing parameters can be adjusted via this file.
You need to generate loss and result files.

### three main functions:
DHPR_Image_Selection.py:  Generating a typical training set from input images automatically following diversity and hardness samples first criteria.
Train_Supervise.py: Realizing the training process for a new training or transfer learning for fine tuning.
Predict_Selected_Dataset.py: Loading the trained model for predcting of new testing image.

### Models:
We include two models for users to test. One is the model trained via our DHPR selection rule, and named DHPR_300.ckpt in the 'checkpoints' file. The other is the model trained via large-scale dataset (1500 images), and named USES_1500.ckpy in the 'checkpoints' file.  

### Test Dataset:
We also include 6 testing images for testing  in the 'image' file under the 'test_dataset' file. The prediction results by our DHPR model and large-scale samples model (USES) are also included in the 'DHPR_300_prediction' file and 'USES_1500_prediction' file under the 'test_dataset' file, respectively. 
The datasets can be accessed via: https://github.com/GTreeSoftware/DB-Enhance/releases/tag/testdata1.

### Software:
We also provide an easy-to-use Neuron Segmentation software GTree with graphics interaction. This Software allows user to load the trained model (DHPR model and USES model) and testing image. Then the predicted neuron probability by the trained model is achieved and can be visualized in the software. The software is accessed via: https://github.com/GTreeSoftware/DB-Enhance/releases/tag/1.

### Training datasets for Brain-scale:
We also publish our typical training datasets with 300 neuronal images from brain-scale neuronal dataset to encourage the study of neuron reconstruction. 
The datasets can be accessed via: https://pan.baidu.com/s/16rp-YSyM3ttitus5h3jvMA.  The code for the datasets and more detailed informations can be provided by email: huangqing@hust.edu.cn. 







