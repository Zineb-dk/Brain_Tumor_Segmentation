import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Brain Tumor Segmentation", page_icon=":brain:", layout="wide")

st.title("Brain Tumor Segmentation Viewer")
st.markdown("Upload NIfTI image files to visualize brain tumor segmentation in multiple views.")

def preprocess_image(image_paths):
    modalities = []
    original_shapes = []
    original_affine = None
    
    for img_path in image_paths:
        img = nib.load(img_path)
        if original_affine is None:
            original_affine = img.affine
            
        img_data = img.get_fdata()
        original_shapes.append(img_data.shape)
        cropped_img = img_data[48:192, 48:192, 5:149] # Crop image
        
        # Z-score normalization
        mean_val = np.mean(cropped_img)
        std_val = np.std(cropped_img)
        normalized_img = (cropped_img - mean_val) / std_val
        
        modalities.append(normalized_img)
    
    # Stack modalities
    stacked_img = np.stack(modalities, axis=3)
    stacked_img = np.expand_dims(stacked_img, axis=0)
    
    return stacked_img, original_shapes, original_affine

# Display a single slice with overlay
def display_slice_with_overlay(slice_data, mask_data, alpha=0.3):
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_data, cmap='gray')
    if mask_data is not None:
        colored_mask = np.zeros((*mask_data.shape, 4)) 
        colored_mask[mask_data == 1] = [1, 0, 0, alpha]
        colored_mask[mask_data == 2] = [0, 1, 0, alpha] 
        colored_mask[mask_data == 3] = [0, 0, 1, alpha] 
        plt.imshow(colored_mask)
    
    plt.axis('off')
    return plt

#  Visualize predictions in different views
def visualize_multiview(original_img, prediction, view_option, slice_idx, modality_idx):
    img_data = original_img[0, :, :, :, modality_idx]
    pred_data = None if prediction is None else np.argmax(prediction[0], axis=3)
    x_dim, y_dim, z_dim = img_data.shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if view_option == 'axial':
        slice_idx = min(slice_idx, z_dim-1)
        img_slice = img_data[:, :, slice_idx]
        mask_slice = None if pred_data is None else pred_data[:, :, slice_idx]
        axes[0].set_title(f'Axial View - Slice {slice_idx}/{z_dim-1}')
    elif view_option == 'sagittal':
        slice_idx = min(slice_idx, x_dim-1)
        img_slice = img_data[slice_idx, :, :]
        mask_slice = None if pred_data is None else pred_data[slice_idx, :, :]
        axes[0].set_title(f'Sagittal View - Slice {slice_idx}/{x_dim-1}')
    elif view_option == 'coronal':
        slice_idx = min(slice_idx, y_dim-1)
        img_slice = img_data[:, slice_idx, :]
        mask_slice = None if pred_data is None else pred_data[:, slice_idx, :]
        axes[0].set_title(f'Coronal View - Slice {slice_idx}/{y_dim-1}')
    
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].axis('off')
    
    if mask_slice is not None:
        axes[1].imshow(img_slice, cmap='gray')
        
        colored_mask = np.zeros((*mask_slice.shape, 4)) 
        colored_mask[mask_slice == 1] = [1, 0, 0, 0.3]   
        colored_mask[mask_slice == 2] = [0, 1, 0, 0.3]  
        colored_mask[mask_slice == 3] = [0, 0, 1, 0.3]  
        
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Overlay')
    else:
        axes[1].imshow(img_slice, cmap='gray')
        axes[1].set_title('No Segmentation Available')
    
    axes[1].axis('off')
    plt.tight_layout()
    return fig

def get_max_slice(img_data, view_option):
    if view_option == 'axial':
        return img_data.shape[2] - 1
    elif view_option == 'sagittal':
        return img_data.shape[0] - 1
    elif view_option == 'coronal':
        return img_data.shape[1] - 1
    return 0

def main():
    st.sidebar.header("Model and Input Configuration")
    
    model_path = st.sidebar.text_input(
        "Path to Saved Model", 
        value='u_net_model.hdf5'
    )
    
    if 'preprocessed_img' not in st.session_state:
        st.session_state.preprocessed_img = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Load model
    try:
        if not st.session_state.model_loaded:
            model = tf.keras.models.load_model(model_path, compile=False)
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
    
    # File uploaders
    st.subheader("Upload Patient Modality Images")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        flair_file = st.file_uploader("Upload FLAIR Image", type=['nii', 'nii.gz'])
    with col2:
        t1ce_file = st.file_uploader("Upload T1CE Image", type=['nii', 'nii.gz'])
    with col3:
        t2_file = st.file_uploader("Upload T2 Image", type=['nii', 'nii.gz'])
    
    # Prediction button
    if st.button("Process Images and Predict"):
        if flair_file and t1ce_file and t2_file:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_paths = []
                for file, suffix in zip([flair_file, t1ce_file, t2_file], 
                                        ['_flair.nii', '_t1ce.nii', '_t2.nii']):
                    filepath = os.path.join(tmpdir, file.name)
                    with open(filepath, 'wb') as f:
                        f.write(file.getbuffer())
                    file_paths.append(filepath)
                
                try:
                    preprocessed_img, original_shapes, original_affine = preprocess_image(file_paths)
                    st.session_state.preprocessed_img = preprocessed_img
                    
                    if st.session_state.model_loaded:
                        with st.spinner('Running prediction...'):
                            prediction = st.session_state.model.predict(preprocessed_img)
                            st.session_state.prediction = prediction
                            
                            pred_argmax = np.argmax(prediction[0], axis=3)
                            unique, counts = np.unique(pred_argmax, return_counts=True)
                            tumor_classes = ['Background', 'Enhancing Tumor', 'Tumor Core', 'Whole Tumor']
                            
                            st.subheader("Prediction Statistics")
                            stats_cols = st.columns(len(unique))
                            for i, (cls, count) in enumerate(zip(unique, counts)):
                                if cls < len(tumor_classes):
                                    with stats_cols[i]:
                                        st.metric(label=tumor_classes[cls], value=f"{count} voxels")
                
                except Exception as e:
                    st.error(f"Processing error: {e}")
        else:
            st.warning("Please upload FLAIR, T1CE, and T2 images.")

    st.header("Visualization")
    
    if st.session_state.preprocessed_img is not None:
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        with viz_col1:
            view_option = st.selectbox(
                "Select View", 
                options=['axial', 'sagittal', 'coronal'],
                index=0
            )
        
        with viz_col2:
            modality_option = st.selectbox(
                "Select Modality", 
                options=['FLAIR', 'T1CE', 'T2'],
                index=1
            )
            modality_idx = {'FLAIR': 0, 'T1CE': 1, 'T2': 2}[modality_option]
        
        max_slice = get_max_slice(st.session_state.preprocessed_img[0, :, :, :, 0], view_option)
        
        with viz_col3:
            slice_idx = st.slider(
                "Select Slice", 
                min_value=0, 
                max_value=max_slice, 
                value=max_slice//2
            )
        
        fig = visualize_multiview(
            st.session_state.preprocessed_img,
            st.session_state.prediction,
            view_option,
            slice_idx,
            modality_idx
        )
        
        st.pyplot(fig)

        st.subheader("Tumor Class Legend")
        legend_cols = st.columns(3)
        with legend_cols[0]:
            st.markdown('<div style="color:red; font-weight:bold;">■ Necrotic and non-enhancing tumor core - NCR/NET (Class 1)</div>', unsafe_allow_html=True)
        with legend_cols[1]:
            st.markdown('<div style="color:green; font-weight:bold;">■ Peritumoral edema - ED (Class 2)</div>', unsafe_allow_html=True)
        with legend_cols[2]:
            st.markdown('<div style="color:blue; font-weight:bold;">■ GD Enhancing Tumor - ET (Class 3)</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()