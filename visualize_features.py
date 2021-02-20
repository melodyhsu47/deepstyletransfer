from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
from keras.layers import *
import numpy as np
import glob 
import os
from os import rename, listdir

model=VGG16(weights='imagenet',include_top=True)
model_extractfeatures=Model(input=model.input,output=model.get_layer('fc2').output)

filenames = glob.glob('/Users/melodyhsu/Desktop/utagawa_hiroshige/*')
features_array = []
filenames_array = []

for filename in filenames:
   
    filenames_array.append(filename)
    img_path=str(filename)

    name=img_path.split('/')[-1]
    img=image.load_img(img_path,target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    print 'making predictions for '+str(name)+'...'
   
    fc2_features=model_extractfeatures.predict(x)
    features_array.append(fc2_features)
    #fc2_features=model_extractfeatures.reshape((4096,1))
    #np.savetxt('/Users/melodyhsu/Desktop/CLPS1950/art_project/fc2folder/fc2'+str(name)+'.txt',fc2_features)

np.save('/Users/melodyhsu/Desktop/embedding/fc2_UH.npy',features_array)
np.save('/Users/melodyhsu/Desktop/embedding/filenames_UH.npy', filenames_array)
