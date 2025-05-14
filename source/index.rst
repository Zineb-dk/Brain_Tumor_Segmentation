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

- ``
   - ``

- ``
- `os`

Pipeline
========

Data Preprocessing
==================

Creating the U-Net Model
========================

Create the custom data generator
================================

Train the U-Net Model 
=====================

Deployment with Streamlit
=========================


