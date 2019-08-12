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
import seaborn as sns
import keras
import gc
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pickle
#%%
data = pd.read_csv("labels.csv")

label_to_index = {'Cargo': 0,
                  'Military': 1,
                  'Carrier': 2,
                  'Cruise': 3,
                  'Tanker': 4}
index_to_label = {v: k for k, v in label_to_index.items()}


#%%
sns.barplot([k for k, v in label_to_index.items()], data["category"].value_counts().sort_index())


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
fig = plt.figure(figsize=(17, 17))
for i in range(1, 17):
    ax1 = fig.add_subplot(4, 4,i)
    ax1.imshow(x_train[i])
    ax1.title.set_text(index_to_label[np.where(y_train[i]==1)[0][0]])
    plt.axis("off")


#%%
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

datagen = ImageDataGenerator(rotation_range=45,
                             height_shift_range=0.3,
                             width_shift_range=0.3,
                             horizontal_flip=True) \
                             .fit(x_train, augment=True, rounds=1, seed=2019)

train_generator = datagen.flow(x_train, y_train, batch_size=8, seed=2019)

epoch_steps = train_generator.n // train_generator.batch_size


#%%
fig = plt.figure(figsize=(17, 8.5))

for i in range(1, 9):
    ax1 = fig.add_subplot(2, 4, i)
    ax1.axis('off')
    ax1.title.set_text(index_to_label[np.where(batch[1][i - 1]==1)[0][0]])
    ax1.imshow(batch[0][i - 1])


#%%
model = keras.models.Sequential([
    keras.applications.Xception(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

gc.collect()
#%%
mcp_save = keras.callbacks.ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5',
                                           verbose=1,
                                           monitor='val_loss',
                                           mode='auto')

reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, mode='min')

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=epoch_steps,
                   epochs=30,
                   callbacks=[mcp_save, reduce_lr_loss],
                   validation_data=[x_valid, y_valid])

f = open('history.pckl', 'wb')
pickle.dump(history, f)
f.close()

# f = open('store.pckl', 'rb')
# obj = pickle.load(f)
# f.close()

model.save("final.h5")