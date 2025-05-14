# After renaming the 355 segmentation mask, we can go ahead and execute this function to process the wholde dataset and save the final results
import os
import re
import glob
import ants
import numpy as np
import nibabel as nib
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

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
        return corrected_img.numpy()  # Return as numpy array immediately

    except Exception as e:
        print(f"ERROR processing {img_path}: {str(e)}")
        return None
    
def z_score_normalization(numpy_arr):
    mean_val = np.mean(numpy_arr)
    std_val = np.std(numpy_arr)
    return (numpy_arr - mean_val) / std_val

def process_mask(mask_data):
    mask_uint8 = mask_data.astype(np.uint8)
    mask_uint8[mask_uint8 == 4] = 3
    return to_categorical(mask_uint8, num_classes=4)

def crop_img(volume):
    return volume[48:192, 48:192, 5:149]  # For 240x240x155 input

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
    
# Main execution
input_path = "/content/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
output_path = "/content/data_Processed/BraTS2020_TrainingData_Processed"

patient_dirs = sorted(glob.glob(f"{input_path}/BraTS*"))
success_count = 0

for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
    if process_patient(patient_dir, output_path):
        success_count += 1

print(f"Successfully processed {success_count}/{len(patient_dirs)} patients")