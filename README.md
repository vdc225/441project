# 441project

 --- Summary
Our goal is to determine if we can predict the voxel patterns ascociated with viewing novel images after training our model on 75% of image embedding–voxel response pairs.

 --- Required packages
os, subprocess, pathlib, shutil, h5py, numpy, pandas, scipy.sparse, sklearn.metrics.pairwise, PIL, torch, tqdm, transformers, sklearn.linear_model, sklearn.decomposition, sklearn.cross_decomposition, sklearn.metrics, xgboost.

 --- Background:
The THINGS-fMRI dataset is part of the broader THINGS initiative, which investigates how the human brain represents visual objects and their associated semantic concepts. In this study, three participants viewed thousands of natural images spanning 720 distinct object concepts (represented by approximately 12 images each) during fMRI imaging.

Functional MRI (fMRI) measures neural activity indirectly using the blood oxygen level-dependent signal, which reflects changes in blood oxygenation associated with neural activity. fMRI divides the brain into 2-3mm cubic pixels called voxels, each representing the aggregated activity of many thousands of neurons.

CLIP embeddings are used to model visual representations. CLIP is a neural network that maps images into a shared high-dimensional feature space. These embeddings capture semantic and visual similarity, making them well-suited for linking image content to brain activity.


 --- Data:
Input data X consists of image embeddings computed for each stimulus presented in the experiment. The images were obtained from the THINGS dataset and aligned with the fMRI recordings using their unique stimulus identifiers.

Output variable Y consists of neural response patterns derived from fMRI data. Rather than working with raw time-series signals, we use precomputed ICA-based beta estimates, which represent the estimated neural response to each individual stimulus after accounting for noise sources such as head motion and physiological artifacts.

For each image, we compute a 512-dimensional embedding using CLIP. These embeddings are then L2-normalized to unit length, ensuring consistent scaling and making similarity comparisons more meaningful. We aggregate repeated presentations of the same image by averaging their corresponding beta responses, resulting in an image-level dataset with 8003 samples (one per unique image) and approximately 200,000 voxel features per subject.

To focus on stimulus-relevant brain activity, we restrict our analysis to voxels within visually responsive regions of the cortex (e.g., early visual areas and higher-level ventral stream regions), as identified through the provided voxel metadata. These regions are expected to exhibit the strongest variation as a function of visual stimulus content, so we isolated them using Partial Least Squares (PLS) supervised dimensionality reduction.

We first applied z-score normalization for each voxel to standardize the response scale. We then apply Principal Component Analysis (PCA) to reduce the dimensionality of the neural data to 200 dimensions, retaining the components that explain the majority of variance. Lastly, we conduct PLS on this reduced data to represent the 20/50 features with the greatest dependency on our X data.

 --- Directions to run
All codes were run through Google Colab due to the large size of the data. All filepaths are set from when the code was run initially to produce the models and statistics discussed in the presentation. To tun this code on your own Google Drive, filwpaths will have to be replaced depending on where you download the data. First download and unzip the zip file. Inside you will find the .ipynb files "main" for replicating the downloading data from the web, computing embeddings from local images, as well as preprocessing and modeling steps for replicating the analysis. There will be a folder "data" containing two additional folder, each of which contain half of the images expected for the CLIP embeddings. After reassiging all filepaths, the "main" script should run from start to end, with the only additional changes required being selecting the desired subject path {1, 2, 3} and whether to use original, PCA, or PLS data in some instances. Due to the high number of repeated similar analyses, we expect you to manually change these yourself for whatever specific results you want to replicate, as an all encompassing script would become extremely long due to the different data forms and high number of models. 

