from tensorflow.keras.layers import Conv2D, Conv3D, Activation, Input, Concatenate
from tensorflow.keras.layers import Reshape, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import UpSampling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K

def ASPP_model_extdata(insize, output_act='sigmoid'):
    inputs = []
    for elem in insize:
        inputs.append(Input(shape=elem))

    x = Conv2D(12, (3, 3), padding='same', activation='relu')(inputs[0])
    x_ = Conv2D(12, (3, 3), padding='same', activation='relu')(x)

    x = BatchNormalization()(x_)
    x = Dropout(0.25)(x)

    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    x1 = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding="same")(x)
    x2 = AveragePooling2D(pool_size=(4,4), strides=(1,1), padding="same")(x)
    x3 = AveragePooling2D(pool_size=(8,8), strides=(1,1), padding="same")(x)
    x4 = AveragePooling2D(pool_size=(16,16), strides=(1,1), padding="same")(x)

    x1 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(2, 2), activation='relu')(x1)    
    x2 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(4, 4), activation='relu')(x2)
    x3 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(8, 8), activation='relu')(x3)
    x4 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(16, 16), activation='relu')(x4)

    temp = []
    for i in range(1,len(insize)):
        temp.append(UpSampling2D(size=(insize[0][0]//insize[i][0],
                                     insize[0][1]//insize[i][1]))(inputs[i]))

    x = Concatenate()([x_, x1, x2, x3, x4]+temp)
    predictions = Conv2D(1, (1, 1), padding='same', activation=output_act)(x)

    model = Model(inputs=inputs, outputs=predictions)

    return(model)

def ASPP_model_extdata_v2(insize, output_act='sigmoid'):
    inputs = []
    for elem in insize:
        inputs.append(Input(shape=elem))

    x = Conv2D(12, (3, 3), padding='same', activation='relu')(inputs[0])
    x_ = Conv2D(12, (3, 3), padding='same', activation='relu')(x)

    x = BatchNormalization()(x_)
    x = Dropout(0.25)(x)

    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    x1 = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding="same")(x)
    x2 = AveragePooling2D(pool_size=(4,4), strides=(1,1), padding="same")(x)
    x3 = AveragePooling2D(pool_size=(8,8), strides=(1,1), padding="same")(x)
    x4 = AveragePooling2D(pool_size=(16,16), strides=(1,1), padding="same")(x)

    x1 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(2, 2), activation='relu')(x1)    
    x2 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(4, 4), activation='relu')(x2)
    x3 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(8, 8), activation='relu')(x3)
    x4 = Conv2D(24, (3, 3), padding='same',
                  dilation_rate=(16, 16), activation='relu')(x4)

    temp = []
    if len(insize)>1:
        for i in range(1,len(insize)):
            temp.append(UpSampling2D(size=(insize[0][0]//insize[i][0],
                                        insize[0][1]//insize[i][1]),
                                        interpolation='bilinear')(inputs[i]))

    x = Concatenate()([x_, x1, x2, x3, x4]+temp)
    x =  Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    predictions = Conv2D(1, (1, 1), padding='same', activation=output_act)(x)

    model = Model(inputs=inputs, outputs=predictions)

    return(model)



