from __future__ import print_function

import time
from PIL import Image
import numpy as np

import keras.backend as K
from keras.models import Model

from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


#set up input image settings
ht = 200
wd = 200
content_image = Image.open('/Users/melodyhsu/Desktop/matterhorn.jpg')
content_image = content_image.resize((ht,wd))
content_image.show()

#set up target image settings
style_image = Image.open('/Users/melodyhsu/Desktop/museum/matisse_12')
style_image = style_image.resize((ht,wd))
style_image.show()
style_image.save('/Users/melodyhsu/Desktop/matisse_12_orig.jpg')


content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)

style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)


#input preprocessing:
#subtract mean of RGB values of image from the content and style images.
content_array[:, :, :, 0] -= 103.939
content_array[:, :, :, 1] -= 116.779
content_array[:, :, :, 2] -= 123.68

style_array[:, :, :, 0] -= 103.939
style_array[:, :, :, 1] -= 116.779
style_array[:, :, :, 2] -= 123.68

#flip ordering from RBG to BGR
content_array = content_array[:, :, :, ::-1]
style_array = style_array[:, :, :, ::-1]

#put arrays into tensors
content_image = K.variable(content_array)
style_image = K.variable(style_array)

#create tensor "x" which represents style-transferred image
combination_image = K.placeholder((1,ht, wd, 3))
input_tensor = K.concatenate([content_image, style_image, combination_image], axis=0)

#VGG16 from Keras
model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
layers = dict([(layer.name, layer.output) for layer in model.layers])

#set up model weights
content_weight = 0 #output will visually look like the content image
style_weight = 5.0 #output will have the "style" of the style image
total_variation_weight = 1.0
loss = K.variable(0.)

def content_loss(content, combination):
    return K.sum(K.square(combination - content))

layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight*content_loss(content_image_features, combination_features)

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = ht * wd
    return K.sum(K.square(S - C))/(4. *(channels ** 2)*(size ** 2))

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

def total_variation_loss(x):
    a = K.square(x[:, :ht-1, :wd-1, :] - x[:, 1:, :wd-1, :])
    b = K.square(x[:, :ht-1, :wd-1, :] - x[:, :ht-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, ht, wd, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, ht, wd, 3)) - 128.

iterations = 20

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


x = x.reshape((ht, wd, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

final_image = Image.fromarray(x)
final_image.save('/Users/melodyhsu/Desktop/StyleTransferredImage.png')
