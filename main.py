import argparse
import tensorflow as tf
from keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

import numpy as np
import time

from utils import input_crop, preprocess_data, jacard_coef, jacard_coef_loss
from model import unet

start_all_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--cropsize', type=list, required=True)
parser.add_argument('--stride', type=list, required=True)
parser.add_argument('--n_classes', type=int, required=True, help='Number of classes in segmentation problem')
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs in training')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
arg_opt = parser.parse_args()

# inputs

cropsize = arg_opt.cropsize  # [256,256]
stride = arg_opt.stride
lr = arg_opt.lr
n_classes = arg_opt.n_classes
num_epochs = arg_opt.epochs
batch_size = arg_opt.batchsize

# load data
data_train = np.load('./dataset/data_train.npz')['data']
label_train = np.load('./dataset/labels_train.npz')['labels']

# pre-process data
crops_img = np.array(preprocess_data(images=np.array(input_crop(data_train, label_train, cropsize, stride)[0])))

# convert label from (1 to 6) to (0 to 5) and expand dims
crops_label = np.array([*map((lambda x:
                              np.expand_dims(x - 1, 2)), input_crop(data_train, label_train, cropsize, stride)[1])])

# Split image in train and val
total = len(crops_img)
split = int(total)

val_image = crops_img[split:int(total * 0.5)]
train_image = crops_img[:split]

val_label = to_categorical(crops_label[split:int(total * 0.5)], num_classes=n_classes)
train_label = to_categorical(crops_label[:split], num_classes=n_classes)

# Define model

y_true = Input(shape=(cropsize[0], cropsize[1], 1), name='y_true', dtype=tf.int8)

model = unet([cropsize[0], cropsize[1], 3], n_classes)
model = Model(inputs=model.inputs, outputs=model.outputs)

optim = tf.keras.optimizers.Adam(lr)

model.compile(optimizer=optim, loss=[jacard_coef_loss], metrics=[jacard_coef])

# run model

loss_history = []
acc_history = []
iou_history = []
n_batches_train = len(train_image) // batch_size
n_batches_val = len(val_image) // batch_size

for epoch in range(num_epochs):
    print('Starting epoch {}-{}'.format(epoch + 1, num_epochs))
    batch_start = 0  # Initial batch value of j epoch
    batch_end = batch_size  # Initial batch end of j epoch
    loss = 0
    iou_train = 0
    previous_loss = 99999  # initial previous_loss to put loss_val
    previous_loss_val = 99999

    start_time = time.time()
    for i in range(n_batches_train):
        history = model.train_on_batch(x=train_image[batch_start:batch_end],
                                       y=train_label[batch_start:batch_end]
                                       )
        # print(history)
        batch_start += batch_size
        batch_end += batch_size
        loss += history[0]
        iou_train += history[1]
    end_time = time.time()
    print("epoch time: ", end_time - start_time)

    loss = loss / n_batches_train  # Average loss function of all batches in j epoch (train)
    iou_train = iou_train / n_batches_train  # Average IOU of all batches in j epoch (train)

    # Val

    batch_start_val = 0
    batch_end_val = batch_size
    loss_val = 0
    iou_val = 0

    for i in range(n_batches_val):
        history_val = model.test_on_batch(x=val_image[batch_start_val:batch_end_val],
                                          y=val_label[batch_start_val:batch_end_val]
                                          )
        batch_start_val += batch_size
        batch_end_val += batch_size
        loss_val += history_val[0]
        iou_val += history_val[1]

    loss_val = loss_val / n_batches_val
    iou_val = iou_val / n_batches_val

    if loss_val < previous_loss_val:
        model.save_weights('diceloss_unet_' + str(cropsize) + "_" + "_nclass_"
                           + str(n_classes) + '_seg_' + 'weights.h5')
        previous_loss_val = loss_val

    if loss < previous_loss:
        model.save_weights('diceloss_unet_' + str(cropsize) + "_" + "_nclass_"
                           + str(n_classes) + '_seg_' + 'weights_best_Train.h5')
        loss_ant = loss

    print("loss train: ", loss)
    print("iou train: ", iou_train)
    print("loss val: ", loss_val)
