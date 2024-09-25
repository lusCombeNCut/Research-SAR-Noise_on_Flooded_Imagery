import sentinel1decoder
import pandas as pd
import numpy as np
import logging
import math
import cmath
import struct
import os as system
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
import cv2

# filepath = ""
# filename = "S1A_IW_RAW__0SDV_20240609T180612_20240609T180644_054250_06993F_39D8\s1a-iw-raw-s-vh-20240609t180612-20240609t180644-054250-06993f-annot.dat"
#filepath = "../data/russia/"
#filename = "s1a-iw-raw-s-vh-20230307t152137-20230307t152158-047540-05b562.dat"

inputfile = "S1A_S3_RAW__0SDH_20240617T213605.SAFE\S1A_S3_RAW__0SDH_20240617T213605_20240617T213630_054368_069D4F_ABA9.SAFE\s1a-s3-raw-s-hh-20240617t213605-20240617t213630-054368-069d4f.dat"
l0file = sentinel1decoder.Level0File(inputfile)

selected_burst = 8
selection = l0file.get_burst_metadata(selected_burst)

# Decode the IQ data
original_radar_data = l0file.get_burst_data(selected_burst, True)
slow_time_truncated = int(original_radar_data.shape[1] * 0.2)

radar_data = original_radar_data[:, :slow_time_truncated]

SNR = 10**(40/10)
print("~~~ New SNR Test ~~~ ", SNR)

def generate_thermal_noise(shape, sigma):
    noise_real = np.random.normal(0, sigma, shape)
    noise_imag = np.random.normal(0, sigma, shape)
    return noise_real + 1j * noise_imag

if SNR != 0:
    mean_value = np.mean(abs(radar_data)**2)
    additional_noise_power = mean_value / SNR
    additional_noise = generate_thermal_noise(radar_data.shape, np.sqrt(additional_noise_power/2))

    # Add noise to radar data
    radar_data += additional_noise

def crop_row(image, amount):
    # Crops the image vertically to focus on a ROI 
    num_cols_to_remove = int(amount * image.shape[0])
    output = image[:num_cols_to_remove, :]
    return output

# # Plot the noisy IQ data
# plt.imshow(crop_row(abs(radar_data[:, :]), 0.25), vmin=0, vmax=15, origin='lower')
# plt.xlabel("Fast Time")
# plt.ylabel("Slow Time")
# plt.axis('off')
# plt.tight_layout()
# plt.show()

len_range_line = radar_data.shape[1]
len_az_line = radar_data.shape[0]

# Tx pulse parameters
c = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS
RGDEC = selection["Range Decimation"].unique()[0]
PRI = selection["PRI"].unique()[0]
rank = selection["Rank"].unique()[0]
suppressed_data_time = 320/(8*sentinel1decoder.constants.F_REF)
range_start_time = selection["SWST"].unique()[0] + suppressed_data_time
wavelength = sentinel1decoder.constants.TX_WAVELENGTH_M

# Sample rates
range_sample_freq = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
range_sample_period = 1/range_sample_freq
az_sample_freq = 1 / PRI
az_sample_period = PRI

# Fast time vector - defines the time axis along the fast time direction
sample_num_along_range_line = np.arange(0, len_range_line, 1)
fast_time_vec = range_start_time + (range_sample_period * sample_num_along_range_line)

# Slant range vector - defines R0, the range of closest approach, for each range cell
slant_range_vec = ((rank * PRI) + fast_time_vec) * c/2
    
# Axes - defines the frequency axes in each direction after FFT
SWL = len_range_line/range_sample_freq
az_freq_vals = np.arange(-az_sample_freq/2, az_sample_freq/2, 1/(PRI*len_az_line))
range_freq_vals = np.arange(-range_sample_freq/2, range_sample_freq/2, 1/SWL)

# Spacecraft velocity - numerical calculation of the effective spacecraft velocity
ecef_vels = l0file.ephemeris.apply(lambda x: math.sqrt(x["X-axis velocity ECEF"]**2 + x["Y-axis velocity ECEF"]**2 +x["Z-axis velocity ECEF"]**2), axis=1)
velocity_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), ecef_vels.unique(), fill_value="extrapolate")
x_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), l0file.ephemeris["X-axis position ECEF"].unique(), fill_value="extrapolate")
y_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), l0file.ephemeris["Y-axis position ECEF"].unique(), fill_value="extrapolate")
z_interp = interp1d(l0file.ephemeris["POD Solution Data Timestamp"].unique(), l0file.ephemeris["Z-axis position ECEF"].unique(), fill_value="extrapolate")
space_velocities = selection.apply(lambda x: velocity_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)

x_positions = selection.apply(lambda x: x_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
y_positions = selection.apply(lambda x: y_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
z_positions = selection.apply(lambda x: z_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)

position_array = np.transpose(np.vstack((x_positions, y_positions, z_positions)))

a = sentinel1decoder.constants.WGS84_SEMI_MAJOR_AXIS_M
b = sentinel1decoder.constants.WGS84_SEMI_MINOR_AXIS_M
H = np.linalg.norm(position_array, axis=1)
W = np.divide(space_velocities, H)
lat = np.arctan(np.divide(position_array[:, 2], position_array[:, 0]))
local_earth_rad = np.sqrt(
    np.divide(
        (np.square(a**2 * np.cos(lat)) + np.square(b**2 * np.sin(lat))),
        (np.square(a * np.cos(lat)) + np.square(b * np.sin(lat)))
    )
)
cos_beta = (np.divide(np.square(local_earth_rad) + np.square(H) - np.square(slant_range_vec[:, np.newaxis]) , 2 * local_earth_rad * H))
ground_velocities = local_earth_rad * W * cos_beta

effective_velocities = np.sqrt(space_velocities * ground_velocities)

D = np.sqrt(
    1 - np.divide(
        wavelength**2 * np.square(az_freq_vals),
        4 * np.square(effective_velocities)
    )
).T

# We're only interested in keeping D, so free up some memory by deleting these large arrays.
del effective_velocities
del ground_velocities
del cos_beta
del local_earth_rad
del H
del W
del lat

# FFT each range line
radar_data = np.fft.fft(radar_data, axis=1)
# FFT each azimuth line
radar_data = np.fft.fftshift(np.fft.fft(radar_data, axis=0), axes=0)

# Create replica pulse
TXPSF = selection["Tx Pulse Start Frequency"].unique()[0]
TXPRR = selection["Tx Ramp Rate"].unique()[0]
TXPL = selection["Tx Pulse Length"].unique()[0]
num_tx_vals = int(TXPL*range_sample_freq)
tx_replica_time_vals = np.linspace(-TXPL/2, TXPL/2, num=num_tx_vals)
phi1 = TXPSF + TXPRR*TXPL/2
phi2 = TXPRR/2
tx_replica = np.exp(2j * np.pi * (phi1*tx_replica_time_vals + phi2*tx_replica_time_vals**2))

# Create range filter from replica pulse
range_filter = np.zeros(len_range_line, dtype=complex)

index_start = np.ceil((len_range_line - num_tx_vals) / 2)-1
index_end = num_tx_vals + np.ceil((len_range_line - num_tx_vals) / 2)-2

range_filter[int(index_start):int(index_end+1)] = tx_replica
range_filter = np.conjugate(np.fft.fft(range_filter))

# Apply filter
radar_data = np.multiply(radar_data, range_filter)

del range_filter
del tx_replica

# Create RCMC filter
range_freq_vals = np.linspace(-range_sample_freq/2, range_sample_freq/2, num=len_range_line)
rcmc_shift = slant_range_vec[0] * (np.divide(1, D) - 1)
rcmc_filter = np.exp(4j * np.pi * range_freq_vals * rcmc_shift / c)

# Apply filter
radar_data = np.multiply(radar_data, rcmc_filter)

del rcmc_shift
del rcmc_filter
del range_freq_vals

radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, axis=1), axes=1)

# Create filter
az_filter = np.exp(4j * np.pi * slant_range_vec * D / wavelength)

# Apply filter
radar_data = np.multiply(radar_data, az_filter)

del az_filter

def normalise(input):
    # Normalise input array to between 0-1
    return (input - input.min()) / (input.max() - input.min())

def otsu_thresholding(image):
    # Performs Otsu Thresholding
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    threshold_value, binary_image = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image, threshold_value

radar_data = np.fft.ifft(radar_data, axis=0)
image_scene = crop_row(np.flipud(np.log10(abs(radar_data) + 1)), 0.25)

# downsample_factor = 0.5
# original_height, original_width = image_scene.shape[:2]
# new_width = int(original_width * downsample_factor)
# new_height = int(original_height * downsample_factor)
# downsampled_image = cv2.resize(image_scene, (new_width, new_height), interpolation=cv2.INTER_AREA)
# image_scene = cv2.resize(downsampled_image, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

# ~~~ Perform thresholding ~~~
S1_CLIPP = 1  # Clip threshold
image_scene = normalise(image_scene)
image_scene_clipped = normalise(np.clip(image_scene, 0, S1_CLIPP))
binary_diff, threshold_value = otsu_thresholding(image_scene_clipped)

#binary_diff = np.where(binary_diff > threshold_value, 1, 0)
binary_diff = np.where(binary_diff==0, 1, 0)
print(threshold_value)

if binary_diff.dtype != np.uint8:
    binary_diff = (binary_diff > 0).astype(np.uint8) * 255

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_diff, connectivity=4)

min_size = 100000  # Minimum Size for CCA to keep area
filtered_map = np.zeros_like(binary_diff)

for label in range(1, num_labels):  # Start from 1 to skip the background component
    if stats[label, cv2.CC_STAT_AREA] >= min_size:
        # If the component's area is greater than or equal to min_size, keep it
        filtered_map[labels == label] = 1

if SNR != 0:
    np.save('filtered_map.npy', filtered_map)
    print("Saving noisy output")
else:
    np.save('filtered_map_noise_free.npy', filtered_map)
    print("Saving clean output")

filtered_map_noise_free = np.load('filtered_map_noise_free.npy')

actual = filtered_map_noise_free.flatten()
predicted = filtered_map.flatten()

tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
conf_matrix = np.array([[tn, fp],
                                        [fn, tp]])

print("Percent correct", 100*(tn+tp)/conf_matrix.sum())
print("Percentage correct when predicted flooded", 100*(tp)/(fp+tp))
print("Percentage flase negatives", 100*(fn)/conf_matrix.sum())
print(conf_matrix)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 200 * (precision * recall) / (precision + recall)
#f1_list.append(f1_score)
print("F1-Score:", f1_score)

plt.subplot(1, 2, 1)
plt.imshow(image_scene, cmap='viridis')
plt.xlabel("Fast Time")
plt.ylabel("Slow Time")
plt.tight_layout()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_map, cmap='grey')
plt.xlabel("Fast Time")
plt.ylabel("Slow Time")
plt.axis('off')
plt.tight_layout()
plt.show()

filtered_map_noise_free = np.load('filtered_map_noise_free.npy')

# plt.title('Noise free')
# plt.imshow(filtered_map_noise_free, cmap='viridis')
# plt.axis('off')
# plt.colorbar()
# plt.show()

# plt.figure(figsize=(5, 3))
# plt.plot(linsapce ,f1_list)
# plt.xlabel('SNR (db)')
# plt.ylabel('F1-score')
# plt.axis('off')
# plt.grid(True)
# plt.show()