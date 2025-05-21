# Brain_Tumor_Segmentation
Implementation of a widely used deep learning architecture (U-Net) for the task of automatic brain tumor segmentation using multi-parametric MRI scans. It uses the BraTS 2020 dataset, which provides annotated MRI images with labels identifying different tumor regions: the necrotic/non-enhancing core, peritumoral edema, and the GD-enhancing tumor.


Check my full project documentaion |Here](https://brain-tumor-segmentation.readthedocs.io/en/latest/). 

This project uses the U-Net deep learning architecture to automatically segment brain tumors from MRI scans. We use the [BraTS2020(Brain Tumor Segmentation) dataset]([https://www.med.upenn.edu/cbica/brats2021/data.html](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)), which contains multi-modal MRI images (T1, T1c, T2, FLAIR) along with expert-labeled ground truth masks.

---
## Requirements
The implementation of this project requires the following:

• Programming Language - Python

• Libraries and Frameworks

- keras :
  
      - keras.models
      - keras.layers
      - keras.optimizers
   
- tensorflow
- tensorflow.keras
- numpy
- sklearn
- splitfolders
- nibabel
- ants (antspyx)
- glob
- tqdm
- matplotlib
- streamlit
- tempfile
- os

## Project Pipeline Overview

Our pipeline consists of the following stages:

1. **Understanding and Preprocessing the Dataset**
   - N4 Bias Field Correction to remove low-frequency intensity non-uniformities using `antsPy`.
   - Z-score intensity normalization for stabilizing training.
   - Spatial cropping to focus on the brain region and reduce memory usage.
   - One-hot encoding of segmentation masks after relabeling.

2. **3D U-Net Architecture**
   - Modified from standard 2D U-Net to work with 3D MRI volumes (144×144×144).
   - Encoder-decoder path with skip connections for high-resolution context recovery.

3. **Training with Custom Data Generator**
   - Generator loads 3D volumes on-the-fly to minimize memory usage.
   - Stratified train/val split (80/20).
   - Trained for 30 epochs using Adam optimizer and categorical cross-entropy loss.

4. **Model Prediction and Visualization**
   - Predicts tumor masks for unseen data using saved `.h5` model.
   - Results visualized for interpretation.

5. **Deployment via Streamlit**
   - A web app was built to let users upload NIfTI scans and get real-time tumor segmentation results.

---

## Project Structure

├── Brain_Tumor_Segmentation_Understanding_Data.ipynb     # Dataset exploration

├── Brain_Tumor_Segmentation_Data_Preprocessing.py        # Preprocessing pipeline

├── Brain_Tumor_Segmentation_U_Net_Training.ipynb         # 3D U-Net training

├── Brain_Tumor_Segmentation_Data_Prediction.ipynb        # Prediction visualization

├── Brain_Tumor_Segmentation_Streamlit.py                 # Streamlit deployment

## Getting Started

### 1. Download and Prepare the Dataset

Download the BraTS 2020 dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).

After downloading:
- Place the dataset in your preferred directory.
- Update any dataset paths in the code.
- Ensure NIfTI support via the `nibabel` library.

### Understanding the BraTS2020 Dataset
- Each patient has 4 MRI modalities: T1, T1ce, T2, FLAIR
- Each scan is 3D: 240×240×155

- Ground truth labels:
  
      - 0: Background
  
      - 1: Necrosis
  
      - 2: Edema
  
      - 4: Enhancing Tumor → remapped to 3

- Format: .nii.gz (NIfTI format)

### 2. Preprocess the Dataset

Run the preprocessing script:

```bash
python Brain_Tumor_Segmentation_Data_Preprocessing.py
```
This will:

- Apply N4 Bias Field Correction using antsPy.

- Perform Z-score normalization (mean = 0, std = 1).

- Crop MRI volumes from 240×240×155 to 144×144×144.

- Relabel segmentation masks (label 4 → 3) and one-hot encode.

NOTE THAT : 🧠 The brain mask ensures the bias field correction is limited to meaningful tissue and not background regions.


## 3. Build and Train the U-Net Model
Now, build and train a U-Net model using the preprocessed data:
```bash
├── Brain_Tumor_Segmentation_U_Net_Training.ipynb
```
### 3D U-Net Architecture
We use a 3D U-Net instead of the traditional 2D U-Net to fully utilize volumetric data.

#### Encoder (Analysis Path)
Composed of 5 convolutional blocks.
- Filters increase as: 16 → 32 → 64 → 128 → 256
- 3D convolutions (3×3×3) with max pooling.
- Compresses input volume from 144×144×144 to 9×9×9 with 256 features.

#### Decoder (Synthesis Path)
- Reconstructs spatial dimensions using transposed 3D convolutions.
- Skip connections concatenate features from the encoder to preserve fine details.
- Final 1×1×1 convolution layer with softmax for 4-class voxel-wise segmentation.

### Training the Model
- Loss Function: Categorical Cross-Entropy (for multi-class segmentation).
- Optimizer: Adam with learning rate 0.001
- Metrics: Accuracy (used with caution due to class imbalance)
- Epochs: 30-
- Batch Size: 2 (due to 3D volume memory constraints)

NOTE THAT : Accuracy is a limited metric in medical segmentation due to class imbalance — Dice score or IoU can be considered for better evaluation.

#### Custom Data Generator
To handle large 3D MRI volumes, we use a custom Python generator that:

- Loads the 4 MRI modalities per subject and stacks them into a multi-channel input.
- Loads the corresponding segmentation mask.
- Yields batches of size 2 for training.

#### Training Strategy
- Stratified random sampling (80% train / 20% validation)
- Training handled via model.fit() with separate generators.
- Model saved in .h5 format after training for easy deployment.

## 4. Model Prediction
Finally, use the following notebook to visualize predictions:
```bash
├── Brain_Tumor_Segmentation_Data_Prediction.ipynb
```

## 5. Streamlit Web App

The trained model is integrated into two separate **Streamlit** web apps:

### `Brain_Tumor_Segmentation_Streamlit.py`
- Performs tumor segmentation after cropping all MRI volumes to a fixed shape of **144×144×144**.
- Suitable when all input scans are preprocessed and standardized.

### `Brain_Tumor_Segmentation_Streamlit_Mask.py`
- Uses **intelligent cropping**: 
  - A brain mask is first generated to isolate relevant brain tissue.
  - Then, a bounding box is computed around the mask to crop the volume tightly around the brain region.
  - This enables **adaptive segmentation** on MRI volumes of varying dimensions.
- Recommended for real-world deployment where scan sizes may vary.

You can run either version locally with:

```bash
streamlit run Brain_Tumor_Segmentation_Streamlit.py
```

Or :
```bash
├── streamlit run Brain_Tumor_Segmentation_Streamlit_Mask.py
```

