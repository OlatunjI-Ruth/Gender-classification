# mounting my drive to google colab
from google.colab import drive
drive.mount("/content/gdrive/")

# specifying the paths for my training and validation dataset
training_dir = "/content/gdrive/MyDrive/Colab Notebooks/gender Dataset/Training"
validation_dir = "/content/gdrive/MyDrive/Colab Notebooks/gender Dataset/Validation"

# importation of important dependencies
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_hub as hub
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# preprocessing and Augumentation of the image dataset
train_gen = ImageDataGenerator(rescale=1./255,
                               horizontal_flip=True,
                               rotation_range=0.2,
                               zoom_range=0.4,
                               shear_range=0.2,
                               width_shift_range=0.4,
                               fill_mode='nearest')
validation_gen = ImageDataGenerator(rescale=1/255)

image_size = 299
Batch_size = 50

train_data_gen = train_gen.flow_from_directory(directory=training_dir,
                                               target_size=(299, 299),
                                               class_mode='binary',
                                               batch_size=Batch_size,
                                               shuffle=True
                                               )

val_data_gen = validation_gen.flow_from_directory(directory=validation_dir,
                                                  target_size=(image_size, image_size),
                                                  class_mode = 'binary',
                                                  batch_size=20
                                                  )

#specifying the link to the TL pretrained model
url = "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4"

#instantiating the model
feature_extractor = hub.KerasLayer(url,
                                   input_shape=(image_size, image_size,3))
#Freezing of the model
feature_extractor.trainable=False

# specifying the last (output) layer of the model
model=tf.keras.Sequential([
    feature_extractor,
    layers.Dense(1, activation='sigmoid')
])

#model compilation
model.compile(optimizer='Adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

#model training
model.fit(train_data_gen,
          epochs=20,
          validation_data=val_data_gen)

#plotting a graph comparison of the training and vallidation loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(Epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()

#making predictions with the model.
predictions = model.predict(validation_data_gen)

predictions[0]



