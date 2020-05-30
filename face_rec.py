import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

#os.chdir('.spyder-py3/tensorflow')

#preprocessing images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        vertical_flip=True,
        height_shift_range=0.2,
        width_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=5,
        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=5,
        class_mode='categorical')

base_model=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3),include_top=False,weights= 'imagenet')

base_model.trainable = False
convolution_layer=tf.keras.layers.Convolution2D(128,3,3, input_shape=(224,224,3),activation='softmax')
pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
prediction_layer = tf.keras.layers.Dense(3)

model = tf.keras.Sequential([
  base_model,
  convolution_layer,
  pooling_layer,
  prediction_layer
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

initial_epochs = 15
history=model.fit(train_generator,
          epochs=initial_epochs,
          validation_data=test_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),2])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

