import os, sys, json
import numpy as np
import pandas as pd
import netCDF4 as nc
from asiplib import sliding_window, maskeddata_to_patches_v2


# This dictionary translates SIGRID3 codes to ice concentrations.
# It will be used to save a dictionary for each scene that translates polygon id's
# in the "polygon_icechart" layer to ice-concentrations.
dct = {'00' : 0, '1' : 0.05, '02' : 0, '10' : 0.1, '20' : 0.2, '30' : 0.3, '40' : 0.4, 
       '50' : 0.5, '60' : 0.6, '70' : 0.7, '80' : 0.8, '90' : 0.9, '91' : 0.95, '92' : 1.0}



###################################
#   User Parameters               #
###################################

# Over-Sampling-Factor in fractions e.g. 0.75 = 75% overlap between patches
osf = None

datapath = "/scratch/dmal/asip/dataset-2/"
outpath = "/scratch/dmal/asip/ds2_nersc/"
rm_swath = 0
amsr_labels = ['btemp_6.9h', 'btemp_6.9v', 'btemp_7.3h', 'btemp_7.3v', 
               'btemp_10.7h', 'btemp_10.7v', 'btemp_18.7h', 'btemp_18.7v', 
               'btemp_23.8h', 'btemp_23.8v', 'btemp_36.5h', 'btemp_36.5v', 
               'btemp_89.0h', 'btemp_89.0v']

nersc='nersc_' #Leave as empty string '' for ESA noise corrections or as 'nersc_' 
               # for the Nansen center noise correction.



###################################
#   Extraction of patches         #
###################################

if not os.path.exists(outpath):
    os.mkdir(outpath)

files = [elem for elem in os.listdir(datapath) if elem.endswith(".nc")]

ws = (800, 800)
file_list = []
filesize = 0
i = 1
j = 0
for filename in files:
    amdata = []
    print("Starting %d out of %d files" % (i, len(files)))

    fil = nc.Dataset(datapath+filename)
    lowerbound = max([rm_swath, fil.aoi_upperleft_sample])
    if amsr_labels and not (amsr_labels[0] in fil.variables):
        f = open(outpath+"/discarded_files.txt", "a")
        f.write(filename+",missing AMSR file"+"\n")
        f.close()
        print("wrote "+filename+" to discarded_files.txt in "+outpath)
    elif ((fil.aoi_lowerright_sample-lowerbound) < ws[0] or
        (fil.aoi_lowerright_line-fil.aoi_upperleft_line) < ws[1]):
        f = open(outpath+"/discarded_files.txt", "a")
        f.write(filename+",unmasked scene is too small"+"\n")
        f.close()
        print("wrote "+filename+" to discarded_files.txt in "+outpath)
    
    ######################
    # Process scene      #
    ######################
    else:
        # Run only patch-function if there is even 1 window
        scene  = filename.split('_')[0]
        if not os.path.isdir(outpath+scene):
            os.mkdir(outpath+scene)

        sshape = fil[nersc+'sar_primary'].shape
        
        if osf:
            ss = (int(((ws[0]*osf)//50)*50), int(((ws[1]*osf)//50)*50))
            rss = (int((ws[0]*osf)//50), int((ws[1]*osf)//50))
        else:
            ss = None
            rss = None

        patches, patch_isnan = maskeddata_to_patches_v2(fil, ws, ss, rm_swath=rm_swath, 
                                                    inc=True, nersc=nersc)

        if amsr_labels:
            for elem in amsr_labels:
                amdata.append(fil[elem][:])
            amdata = np.dstack(amdata).astype(np.float32)
            if osf:
                amdata = amdata[:sshape[0]//ss[0]*(ss[0]//50), 
                                :sshape[1]//ss[1]*(ss[1]//50),:]
            else:
                amdata = amdata[:sshape[0]//ws[0]*(ws[0]//50), 
                                :sshape[1]//ws[1]*(ws[1]//50),:]
            if osf:
                rss = (rss[0], rss[1], amdata.shape[-1])
            am_patches = sliding_window(amdata, (ws[0]//50, ws[0]//50, amdata.shape[2]), ss=rss)
            am_patches = am_patches.reshape([np.prod(am_patches.shape[:3])] + list(am_patches.shape[3:]))
            am_patch_isnan = np.isnan(am_patches).any(axis=(1,2,3))
            
            patch_isnan += am_patch_isnan

            patches = patches[patch_isnan==0,:,:,:-1]
            am_patches = am_patches[patch_isnan==0,:,:,:]

        ### save polygon codes ###
        poly_codes = [el.split(';') for el in fil['polygon_codes'][:]]
        df = pd.DataFrame(data=poly_codes[1:], columns=poly_codes[0])
        df.to_csv(outpath+scene+"/polygon_codes.csv")
        json.dump(dict(zip(df.id.values.astype(str), [dct[el] for el in df.CT.values])), 
                  open(outpath+scene+"/polygon2sic.json","w"))
        
        ### save patch ###       
        for j in range(patches.shape[0]):
            np.save(outpath+scene+"/patch"+str(j)+"_x.npy", patches[j, :, :, :2].astype("float32"))
            np.save(outpath+scene+"/patch"+str(j)+"_y.npy", patches[j, :, :, 4].astype("int8"))
            np.save(outpath+scene+"/patch"+str(j)+"_inc.npy", ((patches[j, 0, :, 3]-26.1)*10).astype("uint8"))
            np.save(outpath+scene+"/patch"+str(j)+"_dist.npy", patches[j, 0, :, 2].astype("int8"))
            np.save(outpath+scene+"/patch"+str(j)+"_amsr.npy", am_patches[j].astype("float32"))
            # np.save(outpath+scene+"/patch"+str(j)+"_lats.npy", np.hstack(lat_list))
            # np.save(outpath+scene+"/patch"+str(j)+"_lons.npy", np.hstack(lon_list))
            
            file_list.append(outpath+scene+"/patch"+str(j))

        
    i += 1

### save a list of all amsr layers included ###
np.save(outpath+"amsr_labels.npy", amsr_labels)
### save a list of all file-IDs created ###
json.dump(file_list, open(outpath+"file_list.json","w"))