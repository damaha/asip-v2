# ASIP Version 2 - Automated downstream Sea Ice Products code repository

Code for the ASIP, AI4ARCTIC projects on sea ice prediction in the arctic from Sentinel-1 images. dataset can be found here: [ASIP-v2](https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134). If you publish on the basis of this repository or the dataset, please cite the following paper if relevant for your work: Malmgren-Hansen, David, et al. "A Convolutional Neural Network Architecture for Sentinel-1 and AMSR2 Data Fusion." IEEE Transactions on Geoscience and Remote Sensing (2020).


### DISCLAIMER: This code project is released as it without guarentees and extensive testing. It is meant to guide and help researchers get started on sea ice modelling with convolutional neural networks and since project funding finished in 2020 there will not be future updates.


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

3. A folder with patches saved in subfolders for each scene is produced

```
#Load one patch
path = #path to root folder of extracted patches
my_dict = json.load(open(path+"20181004T180221/polygon2sic.json"))

x_ = np.load(path+"20181004T180221/patch0_x.npy")
amsr_ = np.load(path+"20181004T180221/patch0_amsr.npy")
y_ = np.load(path+"20181004T180221/patch0_y.npy")
y_ = np.expand_dims(np.vectorize(my_dict.get, otypes=[np.float32])(y_.astype(str)), axis=-1)
```
        
