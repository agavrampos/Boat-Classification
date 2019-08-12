#%%
import pandas as pd
import random
import glob
import pathlib
import os
import cv2
import numpy as np
import sys
import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import gc
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


#%%
data = pd.read_csv("labels.csv")

#%%
label_to_index = {'Cargo': 0, 'Military': 1, 'Carrier': 2, 'Cruise': 3, 'Tankers': 4}
index_to_label = {v: k for k, v in label_to_index.items()}

#%%
IMG_SIZE = 224

def preprocess_image(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0)

#%%
x_train = np.zeros(shape=(len(data.image), IMG_SIZE, IMG_SIZE, 3))
for i, image in tqdm.tqdm(enumerate(data.image)):
    x_train[i] = preprocess_image("images\\" + image)

#%%
y_train = data.category
y_train = np.eye(np.max(y_train))[y_train - 1].astype(int)

#%%
print(x_train.shape)
print(y_train.shape)

#%%
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

BATCH_SIZE = 8

datagen = ImageDataGenerator(rotation_range=45, 
                             horizontal_flip=True, 
                             width_shift_range=0.3, 
                             height_shift_range=0.3, 
                             dtype='float32')
datagen.fit(x_train, augment=True, rounds=1, seed=2019)
train_generator = datagen.flow(x_train, y_train, 
                               batch_size=BATCH_SIZE, seed=2019)

step_size_train = train_generator.n // train_generator.batch_size
#%%
fig = plt.figure(figsize=(10, 10))
batch = next(train_generator)

# print(batch[1])

SIZE = 3

for i in range(8):
    ax1 = fig.add_subplot(SIZE, SIZE, i + 1)
    ax1.axis('off')
    ax1.title.set_text(index_to_label[np.where(batch[1][i]==1)[0][0]])
    ax1.imshow(batch[0][i])

#%%
base_model = keras.applications.Xception(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False)

model = keras.models.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

gc.collect()
#%%
# model.save("model.h5")

# mcp_save = keras.callbacks.ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, mode='min')

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=50,
                #    callbacks=[mcp_save, reduce_lr_loss],
                   callbacks=[reduce_lr_loss],
                   validation_data=[x_valid, y_valid])

# model.save("model.h5")

#%%
