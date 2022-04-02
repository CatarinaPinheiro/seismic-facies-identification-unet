import tensorflow as tf
import keras as K
from keras.layers import Input, BatchNormalization, Dropout, Dense, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

# defines convolutional block

def conv2d_block(input_tensor,n_filters, kernel_size=3, batchnorm=True):
    x = Conv2D(n_filters, (kernel_size,kernel_size),padding='same',\
               kernel_initializer='he_normal')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(n_filters, (kernel_size, kernel_size), padding='same',\
               kernel_initializer='he_normal')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

# defines encoder and decoder block
def encoder_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = conv2d_block(input_tensor, n_filters, kernel_size=kernel_size, batchnorm=batchnorm)
    p = MaxPooling2D((2,2))(x)
    return x,p

def decoder_block(input_tensor, n_filters, skip_features, kernel_size=3, batchnorm=True):
    x = Conv2DTranspose(n_filters, (kernel_size, kernel_size), strides=2, padding="same")(input_tensor)
    x = concatenate([x,skip_features])
    x = conv2d_block(x, n_filters, kernel_size=kernel_size, batchnorm=batchnorm)
    return x

# defines unet architecture
#img_input = Input(input_shape)/ out_channel = number of classes

def unet(img_input_shape, out_channel, n_filters=64, k_size=3, batchnorm = True):
    img_input = Input(img_input_shape)
    
    c1, p1 = encoder_block(img_input, n_filters*1, kernel_size=k_size, batchnorm=batchnorm)
    c2, p2 = encoder_block(p1, n_filters*2, kernel_size=k_size, batchnorm=batchnorm)
    c3, p3 = encoder_block(p2, n_filters*4, kernel_size=k_size, batchnorm=batchnorm)
    c4, p4 = encoder_block(p3, n_filters*8, kernel_size=k_size, batchnorm=batchnorm)
    
    
    c5 = conv2d_block(p4, n_filters*16, kernel_size=k_size, batchnorm=batchnorm)
    
    c6 = decoder_block(c5, n_filters*8, c4, kernel_size=k_size, batchnorm=batchnorm)
    c7 = decoder_block(c6, n_filters*4, c3, kernel_size=k_size, batchnorm=batchnorm)
    c8 = decoder_block(c7, n_filters*2, c2, kernel_size=k_size, batchnorm=batchnorm)
    c9 = decoder_block(c8, n_filters*1, c1, kernel_size=k_size, batchnorm=batchnorm)
    
    outputs = Conv2D(out_channel, 1, activation='softmax')(c9) #multiclass segmentation
    model = Model(inputs= img_input, outputs= outputs, name ='U-Net')
    
    return model