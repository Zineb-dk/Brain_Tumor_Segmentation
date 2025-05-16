.. BrainTumorSegmentation documentation master file, created by
   sphinx-quickstart on Tue May 13 12:42:34 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BrainTumorSegmentation documentation
====================================

This project is made by **Zineb DKIER**.

This documentation details the implementation of a U-Net model for automated brain tumor segmentation using the BraTS 2020 dataset, including preprocessing steps such as normalization, bias field correction, and resizing. The final model is integrated into a Streamlit app for real-time visualization and user interaction.

Table of contents
-----------------
- `Introduction <index.html#id3>`_
   - `Brain Tumors and Medical Imaging <index.html#id4>`_
   - `The Transition to AI in Medical Imaging <index.html#id5>`_
   - `MRI Data Format and Acquisition <index.html#id6>`_
   - `MRI Modalities and Their Role in Brain Tumor Imaging <index.html#id7>`_
- `Dataset Description <index.html#id8>`_
- `Requirements <index.html#id9>`_
- `Pipeline <index.html#id10>`_
- `Data Preprocessing <index.html#id11>`_
- `Creating the U-Net Model  <index.html#id12>`_
- `Create the custom data generator <index.html#id13>`_
- `Train the U-Net Model <index.html#id14>`_
- `Deployment with Streamlit <index.html#id15>`_

Introduction
============

Brain Tumors and Medical Imaging
--------------------------------

Brain tumors, especially gliomas, which are common and highly aggressive tumors that originate from glial cells in the brain, are among the most difficult diseases to diagnose and treat. The World Health Organization (WHO) classifies gliomas into four grades (I to IV), with higher grades indicating more dangerous and faster-growing tumors. Both early and accurate brain tumor detection and a comprehensive understanding of the tumor's size are critical to successful treatment. Doctors can use this data to schedule operations, track the progression of the illness, and evaluate how well treatments are working.

Since MRI (Magnetic Resonance Imaging) scans provide detailed images of the brain's soft tissues without requiring invasive procedures, neurologists mainly use them to look into brain tumors. MRI provides information on the tumor's location, size, and appearance. However, manually analyzing MRI scans requires a significant amount of time and effort. The results could also be influenced by the person performing the analysis. Consequently, there is a growing demand for automatic methods to aid in tumor detection and segmentation.

The Transition to AI in Medical Imaging
---------------------------------------

In the past, doctors and researchers used basic image processing techniques like thresholding, region growing, and edge detection to find brain tumors in MRI scans. While these methods worked to some extent, they often had problems because brain tumors can look very different from one patient to another. Also, MRI images can contain noise or be taken with slightly different settings depending on the hospital, which makes it harder for these traditional techniques to be accurate.

These difficulties have caused researchers to try more advanced computer-based techniques. The use of artificial intelligence (AI), especially deep learning, for medical image analysis has significantly increased in recent years. Convolutional Neural Networks (CNNs) are a common deep learning technique. CNNs have shown remarkable success in recognizing and separating brain tumors, and they have fundamentally changed the way computers interpret images.

The potential of deep learning techniques to extract complex patterns from large image collections makes them highly effective. It helps them in managing the various ways that tumors can show up on MRI scans. Also they work well with data from various kinds of patients and hospitals. These models can generate accurate results quick after training, which is very useful in real-world medical scenarios where quick and trustworthy decisions are essential.

Dataset Description
===================
For this study, we used the Brain Tumor Segmentation (BraTS) 2020 dataset, which is a widely used dataset in the field of brain tumor segmentation. This dataset was primarly collected to organize the BraTS challenge, which has been organized annually since 2012, aims to evaluate state-of-the-art methods for the segmentation of brain tumors in multi-parametric MRI scans.
The BraTS 2020 dataset comprises multi-institutional pre-operative MRI scans from 369 patients with histologically confirmed gliomas (293 high-grade glioblastomas and 76 lower-grade gliomas). For each patient, four MRI modalities are available: T1-weighted, T1-weighted with contrast enhancement, T2-weighted, and FLAIR. 

All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple (n=19) institutions...
All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described both in the BraTS 2012--2013 TMI paper and in the latest BraTS summarizing paper. The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution 1mm^3 and skull-stripped.

To better understand the tumor composition, it is helpful to first become familiar with the labeling system. First, we have the background label (0), which includes the rest of the brain, the background, and the non-affected area. Then there's label 1, which encompasses the Necrotic and Non-Enhancing Tumor Core (NCR/NET). It represents the central part of the tumor, the area where brain tissue has died and has been seriously affected by the tumor. It's where the tumor is centrally present, and the brain is seriously damaged.

Next, we have label 2, which stands for Peritumoral Edema (ED). This is the swelling around the tumor , the area that surrounds the actual tumor core. And finally, there's label 4 for GD-Enhancing Tumor (ET). This represents the active and growing part of the tumor. It's where the tumor is most active and where blood vessels are leaking into the tumor tissue, the region where the tumor has the biggest potential to grow.

The first (NCR/NET) and fourth (ET) labels together typically form what's called the tumor core, these are the actual tumor tissues. These two labels are surrounded by the second one (ED), which is the edema ; the swelling that extends outward from the tumor into the surrounding brain tissue.

To access this dataset, you can visit the official BraTS 2020 dataset on Kaggle at: \url{https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation}

Requirements
============
The implementation of this project requires the following:

**• Programming Language**
- `Python`

**• Libraries and Frameworks**

- `keras :`
   - `keras.models`
   - `keras.layers`
   - `keras.optimizers`
- `tensorflow`
- `tensorflow.keras`
- `numpy`
- `sklearn`
- `splitfolders`
- `nibabel`
- `ants (antspyx)`
- `glob`
- `tqdm`
- `matplotlib`
- `streamlit`
- `tempfile`
- `os`

Pipeline
========

Understanding the Data
======================

First, before embarking on the data preprocessing journey, we must first understand
the data structure and how it is arranged. The dataset we used in this project is the
BraTS2020 (Brain Tumor Segmentation) dataset, which consists of multimodal MRI
scans of brain tumors.

Dataset Structure
-----------------

The dataset is organized into two main folders: The Training Folder and The Validation Folder.

#. Training folder: There are 369 patient folders in the training folder. Each patient folder contains 5 NIfTI files: T1 (T1-weighted MRI scan), T1ce (T1-weighted MRI scan with contrast enhancement), T2 (T2-weighted MRI scan), FLAIR (Fluid Attenuated Inversion Recovery scan), and seg (Ground truth segmentation mask).
#. Validation folder: There are 125 patient folders in the validation folder. Each patient folder contains only 4 NIfTI files (T1, T1ce, FLAIR, T2). No segmentation masks are provided (these will be predicted by our model).

Understanding NIfTI Files with NiBabel
--------------------------------------

To be able to read and write neuroimaging data formats, we are going to use a Python library called NiBabel. It particularly works on NIfTI files (.nii or .nii.gz). It allows us to load these medical images as NumPy arrays, making it easy to analyze and manipulate them using Python’s data tools. It also allows us to access useful information like the image dimensions, metadata, and spatial orientation.

Data Dimensions and Modality's Intensity Range
----------------------------------------------

In our dataset, each MRI scan is a 3D volume with 240×240 pixels in each of the 155 slices. We can obtain this information by accessing the shape of the image data array using the shape attribute:

.. code-block:: python

   import nibabel as nib
   import numpy as np

   # Load a NIfTI file (e.g., T1 modality)
   
   t1_img = nib.load('data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii')
   
   # Convert the NIfTI image data to a NumPy array
   
   t1_data = t1_img.get_fdata()
   
   # Print the 3D shape of the volume
   
   print(t1_data.shape)

   # Output : (240, 240, 155)

Understanding Segmentation Masks
--------------------------------

In this dataset, each patient is associated with a mask that identifies the tumor regions in the brain. This segmentation is performed using four distinct labels:

0: Background/healthy tissue

1: Necrotic and non-enhancing tumor core (NCR/NET)

2: Peritumoral edema (ED)

4: Enhancing tumor (ET)

.. code-block:: python
   # Loading a segmentation mask

   seg = nib.load("data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii").get_fdata()
   
   Unique labels in segmentation mask
   
   unique_labels = np.unique(seg)

   # Output :array([0., 1., 2., 4.])

Patient Distribution
--------------------

The dataset contains a total of 369 patients in the training set, with IDs ranging from 001 to 369:

.. code-block:: python

   base_path = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
   patient_folders = [f for f in os.listdir(base_path) if f.startswith("BraTS20_Training_")]
   
   num_patients = len(patient_folders)
   print(f"Total number of patients: {num_patients}")

   # Output : Total number of patients: 369

   min_num = min(int(f.split('')[-1]) for f in patient_folders)
   max_num = max(int(f.split('')[-1]) for f in patient_folders)
   print(f"Patient IDs range from {min_num:03d} to {max_num:03d}")

   # Output Patient IDs range from 001 to 369

Data Preprocessing
==================

Before using this data for deep learning, a number of preprocessing steps are required
to address common issues such as noise, variability across scans, and intensity non-
uniformity. The preprocessing pipeline used to standardise and get the BraTS images
ready for model training is described in this section.

Bias Field Correction
---------------------

Bias Field is a low frequency signal generated by the scanner inhomogeneities when the scan is taken. It can cause  image intensity variations in an mri image, making image processing like segmentation a complicated task to achieve.

The goal here is to improve the image by removing this signal and make the image more homogeneous in intensity, by having the same intensity value for the same type of tissues.

To perform this correction ,we are going to use the N4 bias field correction to find this unwanted low frequency intensity non-uniformity and remove it.

The difference between the two images ( the original one and the corrected one) is sometimes hard to see, that’s why we tend to plot the bias field obtained.

Mathematical Foundation
_______________________

The algoithm first starts by identifying the areas of intensity variation in the original image, this variation can be sometimes hard to see due to slight intensity changes. These areas either look brighter or darker than they should be. 

We can express the image intensity mathemathicaly using this equation : 

I (the original corrupted image with bias field) = I (actual image, without bias field) * B (the bias field)

What we want to do here is remove the bias field from the equation, but the B value - the bias field corruption - is multiplicative, so , we cannot use simple subtraction

The solution is to convert the bias field variation into an additive form. To do that we are going to apply the logarithmic transformation :

log(I (corrupted image)) = log(I (actual image)) + log (B) 

So now, instead of estimating a multiplicative bias, we now estimate an additive bias. And 
instead of complex intensity variations, the bias field is now a low-frequency function that is easier to model and remove.

Implementation with ANTsPy
__________________________

For bias field correction, we utilize the N4 bias field correction algorithm implemented in the ANTsPy library:

.. code-block:: python
   def bias_field_correction(img_path):
       try:
           filename = os.path.basename(img_path)
           match = re.search(r"BraTS20_Training_(\d+)_(t1|t1ce|t2|flair|seg)\.nii", filename)
           if not match:
               return None
   
           original_img = ants.image_read(img_path)
           mask = ants.get_mask(original_img)
           corrected_img = ants.n4_bias_field_correction(
               original_img,
               mask=mask,
               return_bias_field=False
           )
           return corrected_img.numpy()  # Return as numpy array 
   
       except Exception as e:
           print(f"ERROR processing {img_path}: {str(e)}")
           return None


The **get_mask** function creates a binary mask of the brain region, which helps focus the bias field correction on relevant areas. According to the ANTsPy documentation, this function:

- Computes a binary mask from the input image after thresholding
- Can apply morphological operations to clean up the mask (erosion, component analysis, dilation, and closing)

Intensity Normalization
-----------------------

Normalization helps prevent overfitting and speeds up the learning process in neural networks.

Unnormalized MRI data can have intensity values ranging from very small to very large, leading to an inconsistent distribution. If we feed such data directly into our neural network, it can cause instability. Some neurons will develop very high weights to compensate for small input values, while others will have very small weights to adjust for large input values. This imbalance makes the network unstable and harder to train.

To avoid this, we normalize the input data, ensuring it falls within a certain range. This stabilization helps minimize the cost loss value and allows the loss to decrease and go down as training progresses over multiple epochs.

Z-Score Normalization
_____________________

In our preprocessing pipeline, we are going to apply Z-score normalization (also called standardization) to transform the intensity values of our MRI image to have a mean of 0 and a standard deviation of 1. This way, we are going to keep intensity values centered around 0, making training more stable.

Mathematically, Z-score normalization is expressed as:

X' = (X - μ) / σ

Where:

- X is the original intensity value,
- μ is the mean intensity of the image,
- σ is the standard deviation of the image,
- X' is the transformed intensity value.

The Z-score normalization is implemented as follows:

.. code-block:: python
   def z_score_normalization(numpy_arr):
       mean_val = np.mean(numpy_arr)
       std_val = np.std(numpy_arr)
       return (numpy_arr - mean_val) / std_val

 
Spatial Cropping
----------------

MRI brain scans often contain regions outside the brain that are not relevant for tumor segmentation. Additionally, processing full-sized images requires significant computational resources during training. To optimize resource usage while preserving the important features, we apply spatial cropping to the MRI volumes.

The cropping function extracts a specific region of interest from the original 240×240×155 volume, focusing on the central part of the brain where tumors are typically located:

.. code-block:: python
   def crop_img(volume):
       return volume[48:192, 48:192, 5:149]  # For 240x240x155 input

    
This operation reduces the volume size to 144×144×144, significantly decreasing memory requirements while maintaining the relevant information.

Mask Processing
---------------

Class Conversion
________________
The BraTS dataset contains segmentation masks with labels 0, 1, 2, and 4, where:

- 0 represents background
- 1 represents necrotic and non-enhancing tumor core
- 2 represents peritumoral edema
- 4 represents enhancing tumor

For consistency in model training, we convert the label 4 to label 3 and then apply one-hot encoding:

.. code-block:: python
   def process_mask(mask_data):
       mask_uint8 = mask_data.astype(np.uint8)
       mask_uint8[mask_uint8 == 4] = 3
    
This transformation converts the mask into a categorical format with 4 classes (0, 1, 2, 3), which is more suitable for multi-class segmentation tasks.

Complete Processing Pipeline
----------------------------

The complete pipeline integrates all previously described steps, processing each patient's MRI data across all modalities (T1, T1ce, T2, and FLAIR) and segmentation masks:

.. code-block:: python
   def process_patient(patient_dir, output_dir):
       try:
           patient_id = os.path.basename(patient_dir).split('_')[-1]
           files = {
               't1': glob.glob(f"{patient_dir}/*t1.nii")[0],
               't1ce': glob.glob(f"{patient_dir}/*t1ce.nii")[0],
               't2': glob.glob(f"{patient_dir}/*t2.nii")[0],
               'flair': glob.glob(f"{patient_dir}/*flair.nii")[0],
               'seg': glob.glob(f"{patient_dir}/*seg.nii")[0]
           }
   
           # Process all modalities with cropping
           processed = {}
           for mod in ['t1', 't1ce', 't2', 'flair']:
               corrected = bias_field_correction(files[mod])
               if corrected is not None:
                   # Apply cropping and normalization
                   cropped = crop_img(corrected)
                   processed[mod] = z_score_normalization(cropped)
   
           # Process mask with identical cropping
           mask_data = ants.image_read(files['seg']).numpy()
           cropped_mask = crop_img(mask_data)  # Same crop coordinates
           processed_mask = process_mask(cropped_mask)
   
           # Verify shapes
           assert all(v.shape == (144, 144, 144) for v in processed.values())
           assert processed_mask.shape == (144, 144, 144, 4)
   
           # Save all results
           if len(processed) == 4:
               save_processed_data(output_dir, patient_id, processed, processed_mask)
               return True
           return False
   
       except Exception as e:
           print(f"Error processing {patient_dir}: {str(e)}")
           return False

        
\subsection*{Data Saving and Storage}

The preprocessed data is organized into separate directories for images and masks, maintaining the original BraTS naming convention for easy reference:

.. code-block:: python
   def save_processed_data(output_dir, patient_id, modalities, mask):
       os.makedirs(f"{output_dir}_images", exist_ok=True)
       os.makedirs(f"{output_dir}_masks", exist_ok=True)
   
       for mod in ['t1', 't1ce', 't2', 'flair']:
           if mod in modalities:
               nib.save(
                   nib.Nifti1Image(modalities[mod], np.eye(4)),
                   f"{output_dir}_images/BraTS20_Training_{patient_id}_{mod}.nii"
               )
   
       nib.save(
           nib.Nifti1Image(mask, np.eye(4)),
           f"{output_dir}_masks/BraTS20_Training_{patient_id}_seg.nii"
       )

\subsection*{Execution and Validation}

The complete pipeline is executed over the entire BraTS dataset with progress tracking and validation:

.. code-block:: python
   # Main execution
   input_path = "/content/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
   output_path = "/content/data_Processed/BraTS2020_TrainingData_Processed"
   
   patient_dirs = sorted(glob.glob(f"{input_path}/BraTS*"))
   success_count = 0
   
   for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
       if process_patient(patient_dir, output_path):
           success_count += 1
   
   print(f"Successfully processed {success_count}/{len(patient_dirs)} patients")

Creating the U-Net Model
========================
U-Net architectures' efficient encoding-decoding structure with skip connections has made them fundamental for biomedical image segmentation tasks. The implementation examined in this report is appropriate for volumetric medical imaging data, including MRI scans, because it takes the original 2D U-Net concept to three dimensions. This modification is especially pertinent to tasks involving the segmentation of brain tumours, where precise delineation of tumour regions depends on spatial context in all three dimensions.

Model Architecture
------------------

The 3D U-Net model implemented in this project follows the classic U-Net architecture with a contracting path (encoder) and an expansive path (decoder) connected by skip connections. 
The architecture is modified to be compatible for 3D volumetric data with dimensions 144×144×144 pixels and 3 input channels.

The network structure consists of:
- Contracting Path: Five blocks of dual 3D convolution layers followed by max pooling, with progressively increasing feature maps (16→32→64→128→256)
- Bottleneck: A dual 3D convolution block with 256 feature maps
- Expansive Path: Four blocks of upsampling via 3D transposed convolutions, concatenation with corresponding encoder features via skip connections, and dual 3D convolution layers
- Output Layer: A 1×1×1 3D convolution with softmax activation producing a 4-class probability map

Input and Output Specifications
-------------------------------

Input Dimensions:
_________________

- Shape: (144, 144, 144, 3)
- A 3D volumes with 3 input channels (likely representing different MRI sequences)

Output Dimensions:
__________________

- Shape: (144, 144, 144, 4)
- Activation: Softmax
- A multi-class segmentation map with 4 classes 

Layer Configuration
-------------------

The implementation uses:

- Convolution layers: 3×3×3 kernels with 'same' padding throughout
- Pooling: 2×2×2 max pooling for downsampling
- Upsampling: 2×2×2 transposed convolutions with stride 2
- Dropout: Progressive dropout rates increasing with depth (0.1→0.1→0.2→0.2→0.3)
- Activation: ReLU activation functions for all convolutional layers
- Weight initialization: He normal initialization for improved convergence with ReLU activations

Loss Function and Optimization
------------------------------

We used the Categorical cross-entropy as a loss function as it is compatible for multi-class segmentation, and the Adam Optimizer with a learning rate of 0.001. As for the evaluation metric, we used the accuracy (though this is a limited metric for segmentation tasks).

.. code-block:: python
   from keras.models import Model
   from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
   from keras.optimizers import Adam
   import tensorflow as tf
   
   # Input dimensions (BraTS2020 uses 144x144x144x4)
   img_width = 144
   img_height = 144
   img_depth = 144 # 144 is the 3D img size, and is divisable by 16
   kernel_initializer = 'he_normal'
   
   #Define the input layer
   inputs = Input((img_width, img_height, img_depth, 3), name='input_1', dtype=tf.float32)
   
   
   # Encoder: Contracting path
   # Normalize input by dividing by 255 : Convert the input layer from integers to floating points by devinding each pixel by 255
   # s = Lambda(lambda x: x / 255)(inputs)
   s = inputs
   
   #Contraction path
   c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
   c1 = Dropout(0.1)(c1)
   c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
   p1 = MaxPooling3D((2, 2, 2))(c1)
   
   c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
   c2 = Dropout(0.1)(c2)
   c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
   p2 = MaxPooling3D((2, 2, 2))(c2)
   
   c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
   c3 = Dropout(0.2)(c3)
   c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
   p3 = MaxPooling3D((2, 2, 2))(c3)
   
   c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
   c4 = Dropout(0.2)(c4)
   c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
   p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
   
   c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
   c5 = Dropout(0.3)(c5)
   c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
   
   #Expansive path
   u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
   u6 = concatenate([u6, c4])
   c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
   c6 = Dropout(0.2)(c6)
   c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
   
   u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
   u7 = concatenate([u7, c3])
   c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
   c7 = Dropout(0.2)(c7)
   c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
   
   u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
   u8 = concatenate([u8, c2])
   c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
   c8 = Dropout(0.1)(c8)
   c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
   
   u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
   u9 = concatenate([u9, c1])
   c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
   c9 = Dropout(0.1)(c9)
   c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
   
   # We are going to use the softmax activation function since we have multiple classes ( 4 classes )
   # Output layer (4 classes: 0,1,2,3)
   outputs = Conv3D(4, (1,1,1), activation='softmax')(c9)
   
   model = Model(inputs=inputs, outputs=outputs)
   
   loss=tf.keras.losses.CategoricalCrossentropy()
   
   # Here we are going to use the 'adam' optimizer, it a module that contains a lot of back-propagation algorithms that can train our model
   # The optimizer will try to minimize the loss function"
   # Here, we used categorical crossentropy loss for multi-class classification. Once it finds the minimum of this function, the iterations will stop
   # And to mesure the model performance after training we used the 'accuracy' metric
   model.compile(optimizer=Adam(learning_rate=0.001),
                 loss= loss,
                 metrics=['accuracy'])
   
   model.summary()

Create the custom data generator
================================

Train the U-Net Model 
=====================

Deployment with Streamlit
=========================


