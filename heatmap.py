
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from numpy import expand_dims
from scipy.signal import convolve2d
import numpy as np
import cv2

path_img  = 'dogs.png'
model = VGG16(weights='imagenet', include_top = True)

img = load_img(path_img, target_size=(224, 224))

img = img_to_array(img)

img = expand_dims(img, axis=0)

img = preprocess_input(img)

preds = model.predict(img)
arg_output = np.argmax(preds[0])
print('Prediction : ', decode_predictions(preds, top = 1)[0])
print('N°: ', arg_output)


output_model = model.output[:, arg_output]

last_layer = len(model.layers)

last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(output_model, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])


pooled_grads_value, conv_layer_output_value = iterate([img])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

img = cv2.imread(path_img)


heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)


heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img
cv2.imwrite('dogs_heatmap.jpg', superimposed_img)
