import numpy as np
import pandas as pd
import tensorflow.keras, json

    
class asip_generator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_ids, icethreshold, batch_size, dim=(300,300), return_files=False, 
                 shuffle=True, crops=1):

        'Initialization'
        self.ice_threshold = icethreshold
        self.file_ids = file_ids
        self.nb_samples = len(file_ids)
        self.batch_size = batch_size
        self.files_pr_batch = batch_size // crops
        self.return_files = return_files
        self.shuffle = shuffle
        self.crops = crops
        self.dim = dim

        fil = np.load(file_ids[0]+"_x.npy")
        h, w, c = fil.shape
        self.file_shape = (h, w, c)        
        self.ccrop = (h-self.dim[0])//2, (w-self.dim[1])//2 
        
        fil = np.load(file_ids[0]+"_amsr.npy")
        self.nb_amsr = fil.shape[2]

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return(len(self.file_ids) // self.files_pr_batch)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.files_pr_batch:(index+1)*self.files_pr_batch]

        # Find list of IDs
        fbatch = [self.file_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation_s1(fbatch)

        if self.return_files:
            return( X, y, fbatch)
        else:
            return( X, y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation_s1(self, files_batch):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            X = np.empty((self.batch_size, self.dim[0]//self.sub, self.dim[1]//self.sub, self.file_shape[2])) 
            Y = np.empty((self.batch_size, self.dim[0]//self.sub, self.dim[1]//self.sub, 1))

            # Generate data
            inds = []
            for i, fil in enumerate(files_batch):
                inds_x = np.random.randint(low=0, high=(self.file_shape[0]-self.dim[0])//50, size=self.crops)
                inds_y = np.random.randint(low=0, high=(self.file_shape[1]-self.dim[1])//50, size=self.crops)
                for j in range(self.crops):
                    my_dict = json.load(open("/".join(fil.split('/')[:-1])+"/polygon2sic.json"))
                    
                    x_ = np.load(fil+"_x.npy")[inds_x[j]*50:inds_x[j]*50+self.dim[0],
                                            inds_y[j]*50:inds_y[j]*50+self.dim[1],
                                            :]
                    y_ = np.load(fil+"_y.npy")[inds_x[j]*50:inds_x[j]*50+self.dim[0],
                                            inds_y[j]*50:inds_y[j]*50+self.dim[1]]
                    y_ = np.expand_dims(np.vectorize(my_dict.get, otypes=[np.float32])(y_.astype(str)), axis=-1)

                    if self.ice_threshold:
                        y_ = y_ > self.ice_threshold

                    X[(i*self.crops)+j,], Y[(i*self.crops)+j,] = x_, y_

            return([X], Y)
