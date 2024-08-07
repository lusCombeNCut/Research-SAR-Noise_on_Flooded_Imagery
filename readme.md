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
