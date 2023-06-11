# Single Cell Reconstrucion with fine scale

### Why we develop this pipeline?
Revealing the interplay between the structure and function of neuron cells is of crucial importance to understand the neural circuit formulation and facilitate subsequent analysis of brain information flow. Neuron reconstruction which traces the neuron structure from optical images serves as an essential way to investigate how the brain works at a cell level. Although many methods have been developed for neuron tracing, `accurate neuron tracing for neuron images with complicated structures or fuzzy neurites remains an unsolved difficulty`.Revealing the interplay between the structure and function of neuron cells is of crucial importance to understand the neural circuit formulation and facilitate subsequent analysis of brain information flow. Neuron reconstruction which traces the neuron structure from optical images serves as an essential way to investigate how the brain works at a cell level. Although many methods have been developed for neuron tracing, accurate neuron tracing for neuron images with complicated structures or fuzzy neurites remains an unsolved difficulty. **Therefore, we aim to develop a establish a user-friendly neuron reconstruction pipeline with reliable weak neurite detection ability.**


In this project, we fomulated semi-automatic neuron tracing software for precise 3D neuron reconstruction from 2D image stacks by integrating CNN prediction and Voxel-Scooping Tracing Algorithm. In short, to reconstruct a neuron structure from images stacks, the images would be predicted by the pretrained VoxResNet to generate a probability map, with Voxel-Scooping Algorithm delineating the detailed neuron skeleton on the probability map as shown in project pipeline below. 

![image](https://github.com/Zjx01/Single-Cell-Reconstrucion-with-fine--scale/assets/48267562/387bc90c-d242-4e6d-8ef4-83e1e1758a87)
<!-- ABOUT THE PROJECT -->

### Get Started! 
Below is an example of how you can use the pipelines.
1. Clone the repo
   ```sh
   git clone https://github.com/Zjx01/Single-Cell-Reconstrucion-with-fine--scale.git
   ```
2. Run the Image Predicting with 


<p align="right">(<a href="#top">back to top</a>)</p>



### Reconstruction Result 
**CNN prediction result on optical neuron images**
![image](https://github.com/Zjx01/Single-Cell-Reconstrucion-with-fine--scale/assets/48267562/6db17625-66f7-4fd1-b506-27b64ca99cbc)
The pretrained CNN could generally predicted neuron structure in optical neuron images, even for
the ones with tortured structure (a,b) and heavy noises (c,d), validating its role in noise suppressing and neurite prediction. The red arrow here notes broken neurite fragments in the original image and corresponding probability map.


**Comparison of Neuron Tracing Performance under various intensity**
![image](https://github.com/Zjx01/Single-Cell-Reconstrucion-with-fine--scale/assets/48267562/c4370f75-ad41-4d97-b33a-4ab3dc515c21)
We also examine the effectiveness of our tracing methods under various intensity, as low microscopic intensity is quite common problem in the real world. As shown on the right, when the tracing is performed on the original intensity, all tracing methods present a nice neurite tracing performance on original intensity and clearly delineating the overall neurite structure. But When the intensity decreases, some algorithm gradually show weakness in in the detection in the weak neurite, especially for the branches with tortured shapes, which is pointed out by the red circles, while our algorithm is able to achieve an accurate and complete reconstruction effect. 

<p align="right">(<a href="#top">back to top</a>)</p>

### Implementation System Requirements

### Hardware Requirements:
The deep learning algorithm requires enough RAM and GPU to support the calculation. For optimal performace, we recommenda computer with the following specs: RAM: 16+GB CPU: Intel i7 or better GPU: 1080Ti or better

### Software Requirements:
The package development version is tested on Linux operating systems. The developmental version of the package has been tested on the following systems: Linux: Ubuntu 20.04

### Environment Requirements:
|  package   | version  |
|  ----  | ----  |
| CUDA  | cuda 9.0|
| cudnn  | cudnn 7 |
| Python  | 3.9.9 |
| pytorch | 0.4.1 |
|visdom |  0.1.8.5|
|Numpy |  1.14.5|
|tifffile| 0.15.1 |

<p align="right">(<a href="#top">back to top</a>)</p>

### Functions:
For interactive demos of the functions, please give the file paths that include the training and testing images. You can also adjust some paramters for better training or testing in your own computer. The python file config.py is used for configuration of the packages. Paths and training or testing parameters can be adjusted via this file. You need to generate loss and result files.

### three main functions:
DHPR_Image_Selection.py: Generating a typical training set from input images automatically following diversity and hardness samples first criteria. Train_Supervise.py: Realizing the training process for a new training or transfer learning for fine tuning. Predict_Selected_Dataset.py: Loading the trained model for predcting of new testing image.

### Models:
We include two models for users to test. One is the model trained via our DHPR selection rule, and named DHPR_300.ckpt in the 'checkpoints' file. The other is the model trained via large-scale dataset (1500 images), and named USES_1500.ckpy in the 'checkpoints' file.

