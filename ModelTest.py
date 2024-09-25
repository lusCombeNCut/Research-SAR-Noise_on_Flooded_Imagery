import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to also show warnings

# Suppress CUDA output
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # change this to the specific GPU you want to use
import sys
import numpy as np
import tensorflow as tf
import keras_cv
import matplotlib.pyplot as plt
import tifffile as tiff

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = tiff.imread(img_path)
        images.append(img)
    return images

# Function to normalize image data
def normalize_image(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

# Function to load and preprocess a single image and mask
def load_and_preprocess_image_and_mask(s1_path, dem_path, mask_path):
    s1_images = load_images_from_folder(s1_path)
    dem_images = load_images_from_folder(dem_path)
    masks = load_images_from_folder(mask_path)
    
    if len(s1_images) == 0 or len(dem_images) == 0 or len(masks) == 0:
        raise ValueError("No images found in the provided paths.")
    
    s1_image_resized = tf.image.resize(s1_images[0], (256, 256)).numpy()
    dem_image_resized = tf.image.resize(dem_images[0][..., np.newaxis], (256, 256)).numpy().squeeze(axis=-1)
    mask_resized = tf.image.resize(masks[0][..., np.newaxis], (256, 256)).numpy().squeeze(axis=-1)
    
    # Normalize images
    s1_image_resized = normalize_image(s1_image_resized)
    dem_image_resized = normalize_image(dem_image_resized)
    mask_resized = normalize_image(mask_resized)
    
    input_image = np.stack((s1_image_resized[..., 0], s1_image_resized[..., 1], dem_image_resized), axis=-1)
    return input_image, mask_resized

# Load the model from the .keras file
model_path = "lab.keras"
model = tf.keras.models.load_model(model_path)

# Paths to the images
base_path = 'mmflood'
sample_flood_path = os.path.join(base_path, 'mmflood/EMSR518-0')
s1_path = os.path.join(sample_flood_path, 's1_raw')
dem_path = os.path.join(sample_flood_path, 'DEM')
mask_path = os.path.join(sample_flood_path, 'mask')

# Load and preprocess the image and mask
input_image, actual_mask = load_and_preprocess_image_and_mask(s1_path, dem_path, mask_path)
# Print summary statistics of the segmentation result and loss

print("Input image statistics:")
print(f"Min value: {input_image.min()}")
print(f"Max value: {input_image.max()}")
print(f"Mean value: {input_image.mean()}")
print(f"Unique values: {np.unique(input_image)}")

segmentation_result = model.predict(np.expand_dims(input_image, axis=0))[0, :, :, 0]

# Calculate binary cross-entropy loss for the test image
bce_loss = tf.keras.losses.BinaryCrossentropy()
loss_value = bce_loss(tf.expand_dims(actual_mask, axis=-1), tf.expand_dims(segmentation_result, axis=-1)).numpy()

# Print summary statistics of the segmentation result and loss
print("Segmentation result statistics:")
print(f"Min value: {segmentation_result.min()}")
print(f"Max value: {segmentation_result.max()}")
print(f"Mean value: {segmentation_result.mean()}")
print(f"Unique values: {np.unique(segmentation_result)}")
print(f"Binary cross-entropy loss: {loss_value}")

# Display the original image, the actual mask, and the resulting segmentation
plt.figure(figsize=(20, 10))

plt.subplot(1, 5, 1)
plt.imshow(input_image[:, :, 0], cmap='viridis')
plt.title('Original Image (S1 channel 1)')

plt.subplot(1, 5, 2)
plt.imshow(input_image[:, :, 1], cmap='viridis')
plt.title('Original Image (S1 channel 2)')

plt.subplot(1, 5, 3)
plt.imshow(input_image[:, :, 2], cmap='viridis')
plt.title('Original Image (DEM)')

plt.subplot(1, 5, 4)
plt.imshow(actual_mask, cmap='viridis')
plt.title('Actual Flood Mask')

plt.subplot(1, 5, 5)
plt.imshow(segmentation_result, cmap='viridis')
plt.colorbar()
plt.title('Segmentation Result')

print(segmentation_result)
plt.show()
