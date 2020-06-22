import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from random import random
import random 
from skimage import color
import skimage
from skimage.io import imread, imshow
from skimage.transform import resize


import tensorflow as tf
import os

from starter_code.utils import load_case

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

####################################### Wczytywanie zdjec do PNG #####################################################
def ToPNGfunc(Cases):
    cases = np.linspace(2,Cases-3,Cases-4,dtype=np.int)

    TRAIN_PATH_CT = Path('E:\TOM\dataPNG\TRAIN\CT')
    TRAIN_PATH_MASK = Path('E:\TOM\dataPNG\TRAIN\MASK')
    TEST_PATH_CT = Path('E:\TOM\dataPNG\TEST\CT')
    TEST_PATH_MASK = Path('E:\TOM\dataPNG\TEST\MASK')
#    VALID_PATH_CT = Path('E:\TOM\dataPNG\VALID\CT')
#    VALID_PATH_MASK = Path('E:\TOM\dataPNG\VALID\MASK')
    a=0
    b=0
    train_n =0 
    test_n = 0 
    for id in cases:
    ################################ Ładowanie zbioru treningowego zachowyjąc pewnien % czarnych zdjec
        b=a+b
        volume, segmentation = load_case(id)
        segmentation = segmentation.get_fdata()
        volume = volume.get_fdata()
        (a,y,x) = segmentation.shape

        print(f"loading case for TRAINING: {id} out of {Cases}")
        for index in range(b,a+b,1):
            
            if(len(np.unique(segmentation[index-b]))>1 or (random.randint(0,100))<3):
                
                picture_ct = TRAIN_PATH_CT/'{:05d}.png'.format(index) 
                picture_mask = TRAIN_PATH_MASK/'{:05d}.png'.format(index)
                train_n += 1
                    
                plt.imsave(fname=str(picture_ct), arr=volume[index-b], format='png', cmap='gray')
                plt.imsave(fname=str(picture_mask), arr=segmentation[index-b], format='png', cmap='gray')
   
    ################################ Ładowanie zbioru testowego bez czarnych zdjec
    cases = np.linspace(Cases-2,Cases,3,dtype=np.int)
    for id in cases:
        b=a+b
        volume, segmentation = load_case(id)
        segmentation = segmentation.get_fdata()
        volume = volume.get_fdata()
        (a,y,x) = segmentation.shape
    
        print(f"loading case for TESTING: {id} out of {Cases}")
        for index in range(b,a+b,1):
            
            if(len(np.unique(segmentation[index-b]))>1):
                
                picture_ct = TEST_PATH_CT/'{:05d}.png'.format(index) 
                picture_mask = TEST_PATH_MASK/'{:05d}.png'.format(index)
                test_n += 1
                    
                plt.imsave(fname=str(picture_ct), arr=volume[index-b], format='png', cmap='gray')
                plt.imsave(fname=str(picture_mask), arr=segmentation[index-b], format='png', cmap='gray')  
        
    return train_n, test_n
################################################################################################################
######################################### Preprocessing #######################################################  

#train_N, test_N = ToPNGfunc(9)
'''
train_N = 2416
test_N = 103
'''
siz = 256
chan = 1

##zbiór treningowy##
i=0
training_CT = np.zeros((train_N,siz,siz,1),dtype=np.uint8)
print("trainingCT...")
for filename in os.listdir('E:\TOM\dataPNG\TRAIN\CT'):
        img = imread(os.path.join('E:\TOM\dataPNG\TRAIN\CT',filename))
        if img is not None:
            training_CT[i] = resize(img[:,:,0],(siz,siz,1),preserve_range=True)
            i += 1

i=0
print("trainingMASK...")
training_MASK = np.zeros((train_N,siz,siz,1),dtype=np.bool)
for filename in os.listdir('E:\TOM\dataPNG\TRAIN\MASK'):
        img = imread(os.path.join('E:\TOM\dataPNG\TRAIN\MASK',filename))
        if img is not None:
            training_MASK[i] = resize(img[:,:,0],(siz,siz,1),preserve_range=True)
            i += 1


#imshow(training_MASK[65,:,:,0])

##zbiór testowy##

i=0
print("testCT...")
test_CT = np.zeros((test_N,siz,siz,1),dtype=np.uint8)
for filename in os.listdir('E:\TOM\dataPNG\TEST\CT'):
        img = imread(os.path.join('E:\TOM\dataPNG\TEST\CT',filename))
        if img is not None:
            test_CT[i] = resize(img[:,:,0],(siz,siz,1),preserve_range=True)
            i += 1
            
i=0
print("testMASK...")
test_MASK = np.zeros((test_N,siz,siz,1),dtype=np.bool)
for filename in os.listdir('E:\TOM\dataPNG\TEST\MASK'):
        img = imread(os.path.join('E:\TOM\dataPNG\TEST\MASK',filename))
        if img is not None:
            test_MASK[i] = resize(img[:,:,0],(siz,siz,1),preserve_range=True)
            i += 1

####################################### Sieć ##################################


img_width = siz
img_height = siz
img_channels = 1

#model
inputs = tf.keras.layers.Input(shape=(img_width,img_height,img_channels))
#s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#DOWN
c1 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.2)(c1)
c1 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.2)(c2)
c2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.2)(c5)
c5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)
p5 = tf.keras.layers.MaxPooling2D((2,2))(c5)

#UP
u10 = tf.keras.layers.Conv2DTranspose(128,(3,3), strides=(2,2),padding='same')(c5)
u10 = tf.keras.layers.concatenate([u10,c4])
c10 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u10)
c10 = tf.keras.layers.Dropout(0.2)(c10)
c10 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c10)

u11 = tf.keras.layers.Conv2DTranspose(64,(3,3), strides=(2,2),padding='same')(c10)
u11 = tf.keras.layers.concatenate([u11,c3])
c11 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u11)
c11 = tf.keras.layers.Dropout(0.2)(c11)
c11 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c11)

u12 = tf.keras.layers.Conv2DTranspose(32,(3,3), strides=(2,2),padding='same')(c11)
u12 = tf.keras.layers.concatenate([u12,c2])
c12 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u12)
c12 = tf.keras.layers.Dropout(0.2)(c12)
c12 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c12)

u13 = tf.keras.layers.Conv2DTranspose(16,(3,3), strides=(2,2),padding='same')(c12)
u13 = tf.keras.layers.concatenate([u13,c1], axis=3)
c13 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u13)
c13 = tf.keras.layers.Dropout(0.2)(c13)
c13 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c13)

outputs = tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c13)

model = tf.keras.Model(inputs=[inputs],outputs=[outputs])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
session = tf.compat.v1.Session(config=configuration)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir="logs"),
    tf.keras.callbacks.ModelCheckpoint('model_for_segmenation.h5',verbose=1,save_best_only=True)
]
batch_size=128

with tf.device('/:GPU:0'):
    results = model.fit(x=training_CT,y=training_MASK,epochs=5,validation_split=0.30,callbacks=callbacks,verbose=1) 

# PREDYKCJA
preds_train = model.predict(test_CT,verbose=1)

#imshow(preds_train[30,:,:,0])
np.unique(preds_train)

preds = (preds_train > 0.3).astype(np.bool)

#imshow(preds[2,:,:,0])

eva = model.evaluate(test_CT,test_MASK)
'''
x=random.randint(0,train_N)
plt.subplot(1,3,1)
plt.title(f'CT {x}')
plt.axis('off')
plt.imshow(test_CT[x,:,:,0],cmap = 'gray')
plt.subplot(1,3,2)
plt.title(f'mask {x}')
plt.axis('off')
plt.imshow(test_MASK[x,:,:,0],cmap = 'gray')
plt.subplot(1,3,3)
plt.title(f'predicted mask {x}')
plt.axis('off')
plt.imshow(preds[x,:,:,0],cmap = 'gray')
'''







