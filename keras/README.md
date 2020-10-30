#KERAS code for training of Convolutional Neural Networks and running a pretrained network on new data.


### run_model.py
This script take a pretrained model and apply it on a SAR scene from the ASIP-v2 dataset.

**Example of Use**
``` 
$ python run_model.py --model=model2-esa-1s500-s1 --data=PATH_TO_DATASET/20190405T084543_S1B_AMSR2_Icechart-Greenland-CentralEast.nc --out_path=test_results/
```

### train_model.py
Training a CNN model. There are many parameters to change that alters the model or training process and these must simply be changed early in the script in the `param` dictionary. 
When the model has been trained this script produces a number of different files to save all information necessary to reproduce and use the trained model. These files are the following:

* modelname.h5 - This file store the weights of the trained CNN.

* modelname_config.json - This files store the configuration (architecture) of the CNN.

* modelname_history.json - This files stores several information about the training procedure and history (train- and validation accuracy/loss).

* modelname.log - In this file loss information is appended during training so one can follow how the training of the model progress. When the training is done the same information is available in modelname_history.json.

**Example of Use**
``` 
$ python train_model.py
```

### generator_v2.py
Contains a generator function for the dataset exported by `../build_dataset_v2.py` to be used with the 

**Example of Use**
``` 
training_generator = asip_generator(
                        file_ids,           # List of files
                        icethreshold,       # between 0,1 for thresholding the sea ice charts for binary masks, or `None` for returning the sea ice concentration.
                        batch_size,         # Number of samples per batch
                        dim=(300,300),      # Size 
                        return_files=False, # If True, returns filenames for each batch
                        shuffle=True,       # Shuffle file list. 
                        crops=4,            # Number of crops to return per image file in the file_ids)
```