# asip-v2
Code for the ASIP, AI4ARCTIC projects on sea ice prediction in the arctic from Sentinel-1 images.



# ASIP Version 2 - Automated downstream Sea Ice Products code repository

Code for the ASIP, AI4ARCTIC projects on sea ice prediction in the arctic from Sentinel-1 images. dataset can be found here: [ASIP-v2](https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134)

**Dependencies: Python libraries needed to run this code**

* netCDF4

* json

* numpy 

* pandas

### asiplib.py
A library with some functions that is used by several of the other scripts.

**Example of Use**

```
import asiplib as asip
patches, patch_isnan, aoi = asip.maskeddata_to_patches_v2(fil, ws, ss=None, rm_swath=0, inc=False, nersc='') # "fil" is a file object returned by netCDF4.Dataset, "ws" is the window size, "osf" an oversampling factor, rmswath: remove first x lines of scenes, inc:return incidence angle, nersc:nansen center noise correction. 
```

### build_dataset_v2.py
A function that extracts patches of valid data from the scenes

**Example of Use**
1. fill out the user-parameters in the top of the script.

2. `python build_dataset_v2.py` 
