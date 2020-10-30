    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 19:15:41 2017

@author: dmal
"""

import os, json, sys
import numpy as np
import pandas as pd
sys.path.append('..')
from asiplib import sliding_window
from scipy.special import logit

from generator_v2 import asip_generator

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import model_from_json
from models import *
from tensorflow.python.client import device_lib

def jsonize(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[jsonize(k)] = jsonize(v)
        return obj
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [jsonize(i) for i in obj]
    elif type(obj) in (str, int, float, bool, type(None)):
        return obj
    elif callable(obj):
        return str(obj)
    elif hasattr(obj, 'item'):
        # np.int32, np.uint8, np.float32, np.float64, ...
        return obj.item()
    else:
        print("--- UNKNOWN TYPE %s ---" % type(obj))
        raise Exception('unserializable: %s' % type(obj))
    return obj

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

### Parameters ###
param = {"modelname" : "model2-dtu-400", 
         "modeltype" : ASPP_model_extdata,
         "batchsize" : 16,
         "epochs" : 40,
         "period_weights" : 10,
         "seed" : 18374,
         "ice_threshold" : None,
         "path" : "", ### PATH TO DATASET ###
         "dim" : (400,400), 
         "crops" : 4, 
         "nersc" : False,
         "loss_func" : "binary_crossentropy",
         "metrics" : ['acc'],
         "optimizer" : "Adam",
         }
np.random.seed(param['seed'])

files = json.load(open(param['path']))

scenes = [f.split('/')[-2] for f in files]
test_s = np.random.choice(np.unique(scenes), size=int(.2*len(np.unique(scenes))), replace=False)
test_train_ind = [s in test_s for s in scenes]

test_files = np.array(files)[test_train_ind]
train_files = np.array(files)[np.invert(test_train_ind)]

train_generator = asip_generator(train_files, None, param["batchsize"], 
                                 dim=param["dim"], 
                                 crops=param["crops"], sub_f=param["sub_f"], 
                                 nersc=param["nersc"], extdata=param["extdata"])
test_generator = asip_generator(test_files, None, param["batchsize"], 
                                 dim=param["dim"], 
                                 crops=param["crops"], sub_f=param["sub_f"], 
                                 nersc=param["nersc"], extdata=param["extdata"])
nb_trainsamples = train_generator.nb_samples
nb_testsamples = test_generator.nb_samples
batch = test_generator.__getitem__(0)

if param['loss_func'] == "mean_squared_error":
    act = None
else:
    act = 'sigmoid'

model = param['modeltype']([elem.shape[1:] for elem in batch[0]], output_act=act)
del(batch)

model.compile(loss=param['loss_func'], optimizer=param['optimizer'], metrics=param['metrics'])
json_string = model.to_json()
json.dump(json_string, open(param['modelname']+"_config.json", "w"))
model_object = model

#####################################
# Train Model                       #
#####################################

### Use multiple GPUs if available 
nb_gpus = len(get_available_gpus())
if nb_gpus > 1:
    m_model = multi_gpu_model(model, gpus=nb_gpus)
    m_model.compile(loss=param['loss_func'], optimizer=param['optimizer'], metrics=param['metrics'])
    model_object = m_model


import time
start = time.time()
history = model_object.fit_generator(train_generator,
                                     steps_per_epoch=np.ceil(nb_trainsamples//param['batchsize']).astype(int),
                                     callbacks=[ModelCheckpoint("tmp/"+param['modelname'].split('/')[-1]+".{epoch:02d}.h5",
                                                                monitor='loss',
                                                                verbose=1,
                                                                period=param['period_weights'],
                                                                save_weights_only=True),
                                                CSVLogger(param['modelname']+'.log'),
                                                ReduceLROnPlateau(monitor='loss', 
                                                                  factor=0.2,
                                                                  patience=5, 
                                                                  min_lr=0.0001,
                                                                  min_delta=0.001)],
                                     validation_data=test_generator, 
                                     validation_steps=np.ceil(nb_testsamples//param['batchsize']).astype(int),
                                     epochs=param['epochs'], 
                                     max_queue_size=100, 
                                     verbose=1, 
                                     workers=5,
                                     use_multiprocessing=True)
seconds = time.time()-start
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
d, h = divmod(h, 24)
print('It took %d:%d:%02d:%02d to train.' % (d, h, m, s))

model.save_weights(param['modelname']+".h5")
import json
import socket
dct = history.history

dct['loss'] = json_hist['loss']+dct['loss']
dct['val_loss'] = json_hist['val_loss']+dct['val_loss']

dct["number_of_gpus"] = nb_gpus
dct["hostname"] = socket.gethostname()
dct["training_time"] = '%d:%d:%02d:%02d' % (d, h, m, s)
dct["training_files"] = np.unique(["/".join(f.split("/")[:-1]) for f in train_files])
dct["test_files"] = np.unique(["/".join(f.split("/")[:-1]) for f in test_files])
dct.update(param)

json.dump(jsonize(dct), open(param['modelname']+"_history.json", 'w'))
