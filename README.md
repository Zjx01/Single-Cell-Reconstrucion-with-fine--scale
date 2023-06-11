# Single Cell Reconstrucion with fine scale

<!-- ABOUT THE PROJECT -->
### Project overview
![image](https://github.com/Zjx01/Single-Cell-Reconstrucion-with-fine--scale/assets/48267562/933fdca2-be8e-4618-9339-b6b6848cfe8f)


### Why we develop this pipeline?
It is widely known that proper neural function is the physiological basis of our daily behaviors and is tightly regulated by neuron morphology. The distinct arborization structures of neurons are found to underpin their identity, connectivity, and firing pattern, further boosting the formulation of neuron circuits and proper brain functions (Peng et al., 2015). Accumulating studies have observed the neuron deformation and dysconnectivity in patients with degenerative diseases, such as Alzheimer ‘s and Parkinson’s Diseases, indicating the involvement of morphologies changes under disease progression. With such prior knowledge, neuron tracing or neural reconstruction, which extract the neuron structure from microscopic images, is considered as the essential step for neural circuit building and brain information flow analyzing, supporting understanding of pathogenesis of disease.


However, `current methodologies for neuron reconstruction still largely rely on the manual delineation and annotation`, which is labor-intensive, time-consuming. Even a well-experienced expert can take up to days to delineate a single neuron. The lack of effective automatic neuron reconstruction tools  greatly limits the large-scale neuron morphology characterization and quantification and interferes with the downstream neuron circuitry analysis and functional study. Tracing of neuron skeletons from microscopic images was challenged by various factors. The very first difficulty lies in the complicated and distorted dendritic structure of neuron itself. Also, the uneven florescence marker distribution within the neuron cells contribute to the heterogenous neurite intensity and result in the discontinuity and broken shape of neurites, making neuron tracing more difficult. Moreover, various artifacts and noise points caused by different imaging techniques can lead to the over-trace of non-existing neurites from the background and interrupt the accurate delineation of neuron structures.

Despite the difficulties, scientists had developed a series of semi-automatic or automatic neuron reconstruction software and tracing algorithms, including gray-scale weighted distance transform (GWDT) based tracing algorithm APP2, principle curve tracing derived softeware NeuroGPSTree, and open-curve sanke model-based algorithm OpenSnake. But these methods still show `deficiency in weak neurite detection and reqire input of many sensitive parameters`, holding a high demand of algorithm understanding for users. **Therefore, we aim to develop a establish a user-friendly neuron reconstruction pipeline with reliable weak neurite detection ability.**


### Dependencies Requierd to implement the pipeline

### Getting Started:
1. This is an example of how you can implement the pipeline to generate the 
