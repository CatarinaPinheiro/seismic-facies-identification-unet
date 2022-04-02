import skimage.color
from keras import backend as Kb
import numpy as np

def input_crop(data, labels, cropsize, stride):

    img_patches = []
    label_patches = []

    for depth in range(0, data.shape[0]):
        img = data[depth, :, :]
        label = labels[depth, :, :]

        n_crops_x = int(((img.shape[0] - cropsize[0])/stride[1]) + 1) # similar to feature map calculation with zero padding
        n_crops_y = int(((img.shape[1] - cropsize[1])/stride[1]) + 1)
        for i in range(0, n_crops_x):
            for j in range(0,n_crops_y):
                img_patches.append(img[i*stride[0] : cropsize[0]+(i*stride[0]), j*stride[1] : cropsize[1]+(j*stride[1])])

                #if  i == n_crops_x-1 and cropsize[0]+(i*stride[0]) < img.shape[0]:
                #    img_patches.append(img[img.shape[0] - cropsize[0]: img.shape[0], j*stride[1] : cropsize[1]+(j*stride[1])])
                #if  j == n_crops_y-1 and cropsize[1]+(j*stride[1]) < img.shape[1]:
                #    img_patches.append(img[i*stride[0] : cropsize[0]+(i*stride[0]), img.shape[1] - cropsize[1]: img.shape[1]])


                label_patches.append(label[i*stride[0] : cropsize[0]+(i*stride[0]), j*stride[1] : cropsize[1]+(j*stride[1])])

                #if  i == n_crops_x-1 and cropsize[0]+(i*stride[0]) < label.shape[0]:
                #    label_patches.append(label[label.shape[0] - cropsize[0]: label.shape[0], j*stride[1] : cropsize[1]+(j*stride[1])])

                #if  j == n_crops_y-1 and cropsize[1]+(j*stride[1]) < label.shape[1]:
                #    label_patches.append(label[i*stride[0] : cropsize[0]+(i*stride[0]), label.shape[1] - cropsize[1]: label.shape[1]])
                    
    return img_patches, label_patches


def preprocess_data(images):  # images = a list with all arrays (train and test images)
    # Normalize values 0 to 255.0
    normalized_images = list(map(lambda x: (x * 255.0) / x.max(), images))
    colors_images = list(map(lambda m: convert_1_to_3_channel(m), normalized_images))
    return colors_images


def convert_1_to_3_channel(input):
    rgb_input = []
    for i in range(0, len(input)):
        rgb_input.append(skimage.color.gray2rgb(input[i]))
    return np.array(rgb_input)


def jacard_coef(y_true,y_pred):
    y_true = Kb.flatten(y_true)
    y_pred = Kb.flatten(y_pred)
    intersection = Kb.sum(y_true*y_pred)
    return (intersection + 1.0)/(Kb.sum(y_true) + Kb.sum(y_pred) - intersection + 1.0)

def jacard_coef_loss(y_true,y_pred):
    return 1 - jacard_coef(y_true, y_pred)