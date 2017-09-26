import numpy as np

# np.random.seed(1337)  # for reproducibility

import keras
from keras import metrics
from keras import backend as K
from keras import callbacks
from keras import models

from glob import glob

import pickle as pkl

from natsort import natsorted

import h5py as h5

import sklearn.metrics as sklm

import re
import os

#  from ipdb import set_trace as st

from scipy.misc import imsave

img_width, img_height = 28, 28

# input image dimensions
img_rows, img_cols = 28, 28

step = 1


def deprocess(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x=np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_filters(output, number_filters, input_vector):
    filters = []
    bases = []
    for filter_index in range(0, number_filters):
        print 'Processing filter %d...' % filter_index
        loss = K.mean(output[:, :, :, filter_index])
        grads = K.gradients(loss, input_vector)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([input_vector], [loss, grads])

        input_vector_data = np.zeros((1, img_rows, img_cols, 1)) + 128 #np.random.random((1, img_rows, img_cols, 1)) * 20 + 128
        base = np.copy(input_vector_data[0])
        #base = np.reshape(base, (img_height, img_width)) 
        base = deprocess(base)
        bases.append(base)
        #st()
        for i in range(1000):
            loss_value, grads_value = iterate([input_vector_data])
            input_vector_data += grads_value * step
        img = input_vector_data[0]
        #st()
        #img = np.reshape(img, (img_height, img_width))
        img = deprocess(img)
        filters.append(img)
    return (bases, filters)


def extract_kernels(layer):
    weights = layer.get_weights()
    num_kernels = weights[0].shape[3]
    extracted_kernels = []
    for idx in range(num_kernels):
        extracted_kernels.append(
            deprocess(weights[0][:,:,:,idx])
        )

    return extracted_kernels


def save(n_across, n_down, img_width, img_height, margin, arr, filename):
    width = n_across * img_width + (n_across - 1) * margin + 2
    height = n_down * img_height + (n_down - 1) * margin + 2
    stitched_filters = np.zeros((height, width))
    for i in range(0, n_down):
        for j in range(0, n_across):
            idx = i * n_across + j
            img = arr[idx]
            stitched_filters[
                    1 + (img_height + margin) * i : 1 + (img_height + margin) * i + img_height,
                    1 + (img_width + margin) * j : 1 + (img_width + margin) * j + img_width
            ] = img[:, :, 0]
    imsave(filename, stitched_filters)

def load_model():
    checkpoints = glob('checkpoints/*.h5')
    checkpoints = natsorted(checkpoints)
    checkpoint_file = checkpoints[-1]

    model = models.load_model(checkpoint_file)

    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=[metrics.binary_accuracy])

    return model

def save_kernels1(model):
    l = model.get_layer('conv2d_1')
    k = extract_kernels(l)
    save(8, 4, 3, 3, 1, k, 'k0.png')

def save_kernels2(model):
    l = model.get_layer('conv2d_2')
    k = extract_kernels(l)
    save(8, 8, 3, 3, 1, k, 'k1.png')

def main():
    model = load_model()
    print(model.summary())

    layers = dict([(layer.name, layer) for layer in model.layers])
    input_vector = model.input

    bases_0, filters_0 = process_filters(layers['conv2d_1'].output, 32, input_vector)
    bases_1, filters_1 = process_filters(layers['conv2d_2'].output, 64, input_vector)

    save(8, 4, img_width, img_height, 1, bases_0, 'b0.png')
    save(8, 8, img_width, img_height, 1, bases_1, 'b1.png')
    save(8, 4, img_width, img_height, 1, filters_0, 'f0.png')
    save(8, 8, img_width, img_height, 1, filters_1, 'f1.png')

    kernels_0 = extract_kernels(layers['conv2d_1'])
    kernels_1 = extract_kernels(layers['conv2d_2'])

    save(8, 4, 3, 3, 1, kernels_0, 'k0.png')
    save(8, 8, 3, 3, 1, kernels_1, 'k1.png')


if __name__ == "__main__":
    main()