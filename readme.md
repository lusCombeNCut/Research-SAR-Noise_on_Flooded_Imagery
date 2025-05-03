#### Water Segmentation on Elevated Noise SAR Imagery - Applications for Flood detection
Please see the preview.pdf for the first 3 pages of this repositories associated research letter.

This repo contains a range of scripts and notebooks I used and developed to conduct numerical experimentation on the effects of increased noise on SAR satellite imagery, specifically the effect on water segmentation models (both machine learning and thresholding techniques)

##### Data
All data used in this project was gathered from ESA's [copernicus browser](https://browser.dataspace.copernicus.eu/).
! Warning - the SAR level-0 image files can be very large, try and pick a small sample to test frist

Experiment.py - Experimentation script that takes in a SAR image and returns the segmennted image and associate F1 score.
ModelTest.py - Tests the model against the MMfloods dataset.
MMfloods.py - Trains the model on the MMfloods dataset.

Changes to the original sentinel1Level0Decoding notebook have been made from section 5 (image segmentation) and in section 3.2 (Guassian Noise addition).

I have use open-cv and rasterio for image processing, install commands:
```
pip install opencv-python
```
and 
```
conda install -c conda-forge rasterio
```
It seems conda is the easiest way to install rasterio, the code could be adjusted to use tifffile which has simimlar functionality
