# 441project

Summary
Our goal is to determine if we can predict the voxel patterns of new images after training our model on 75% of image embedding–neural response pairs

Background
	Our data was collected from 3 participants viewing a series of 12 images within the same image concept (meaning 36 trials per concept across all participants). A total of 720 unique concepts were shown, usually with 12 images each, although some have 24.

Data
	Our X matrix is the semantic embeddings that represent the concepts of the images that participants viewed during the fMIR. We used CLIP embeddings, as they are trained on word-image pairs, which should generalize well to our dataset. Each concept (e.g., dog, house, candle, tree) was converted into 512 row feature vector.
	Our Y matrix is the voxel patterns measured in response to viewing the images. The regions these voxels are recorded from are ICA-betas derivatives; rather than the full-brain fMRI, we are focusing on select voxels in the visual cortex and nearby object-selective regions. Each voxel representation is a 9840 row vector, covering the most relevant regions that would be affected by viewing an image.

Goal
	We want to determine if training a model on 75% of image embedding–voxel response pairs will allow us to predict the neural patterns of the remaining 25% of categories on which our model will not be trained.
