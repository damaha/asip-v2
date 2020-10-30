#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 17:45:41 2017

@author: dmal
"""

import os, sys, json
sys.path.append('..')
import matplotlib
matplotlib.use('Agg')
from matplotlib.image import imsave
import numpy as np
import netCDF4 as nc
import asiplib as asip
from tensorflow.keras.models import load_model, model_from_json

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-m", "--model", dest="model",
                  help="path to cnn model.")
parser.add_option("-d", "--data", dest="data",  
                  help="path to S1 scene")
parser.add_option("-o", "--out_path", dest="opath", default="",
                  help="path to output dir, default is working dir.")
parser.add_option("-f", "--format", dest="format", default=".tif",  
                  help="format of saved iamge, default .tif")
(options, args) = parser.parse_args()

#####################################
# Prepare Data                      #
#####################################
modelname = options.model
filename = options.data

if "nersc" in modelname:
    nersc = 'nersc_'
else:
    nersc = ''

discard_edge = 50

json_str = json.load(open(modelname+"_config.json","r"))
model = model_from_json(json_str)
model.load_weights(modelname+".h5")

if len(model.input_shape) == 2:
    img_size = (model.input_shape[0][1],model.input_shape[0][2], model.input_shape[0][3])
else:
    img_size = (model.input_shape[1],model.input_shape[2], model.input_shape[3])

fil = nc.Dataset(filename)

sshape = fil[nersc+'sar_primary'].shape
conf = model.layers[0].get_config()
ws = img_size[:2] #window size
ss = (ws[0]-2*discard_edge, ws[1]-2*discard_edge) 
rss = [ss[0]//50, ss[1]//50]

patches, patch_isnan = asip.maskeddata_to_patches_v2(fil, ws, ss=ss, nersc=nersc)
data = patches[patch_isnan==0,:,:,:].copy()
data = data[:,:,:,:2]
ns = data.shape[0]
p_shape = patches.shape
del(patches)

#####################################
# Predict                           #
#####################################
preds = model.predict([data])

ma_image = np.ones(p_shape[:-1]+(1,))*np.nan
ma_image[patch_isnan==0,:,:,-1:] = preds

ma_image = ma_image.reshape(((fil['sar_primary'].shape[0]-2*discard_edge)//(ws[0]-2*discard_edge), 
                            (fil['sar_primary'].shape[1]-2*discard_edge)//(ws[1]-2*discard_edge),                        
                            ws[0],
                            ws[1],
                            1))

# #####################################
# # Reconstruct format and save       #
# #####################################
lst_pre = []
for i in range(ma_image.shape[0]):
    lst_pre.append(np.hstack(ma_image[i,:,discard_edge:-discard_edge,discard_edge:-discard_edge,:]))

ma_image = np.vstack(lst_pre)

scene_shell = np.ones((fil['sar_primary'].shape[0],fil['sar_primary'].shape[1],ma_image.shape[2]))*np.nan
scene_shell[discard_edge:ma_image.shape[0]+discard_edge, discard_edge:ma_image.shape[1]+discard_edge] = ma_image

scene_code = filename.split("/")[-1][:-8]

preds = scene_shell[:,:,-1]*100
isnans = np.isnan(preds)

gcps = asip.get_gcps(fil, aoi=[])

preds[isnans] = 255
preds = np.uint8(preds)
if options.format==".tif":
    asip.save_to_tiff(options.opath+modelname.split("/")[-1]+"_on_"+scene_code+"_predictions.tif",
                    preds,
                    gcps,
                    etype=1,
                    no_data_value=255)
else:
    imsave(options.opath+modelname.split("/")[-1]+"_on_"+scene_code+"_predictions.png", preds)