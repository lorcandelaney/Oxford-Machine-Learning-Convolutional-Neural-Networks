import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print("\n\n\n\n\n")

Ntrain, Nlabels = mnist.train.images.shape

reshaped_train_images = np.zeros(shape=(Ntrain, 28, 28, 1))
for i in range(Ntrain):
	reshaped_train_images[i] = mnist.train.images[i].reshape(28, 28, 1)

'''
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(reshaped_train_images[i].reshape(28, 28), cmap='Greys_r')
plt.show()
'''

model = keras.Sequential([
    keras.layers.Conv2D(filters=25, kernel_size=(12, 12), strides=(2, 2), padding='valid', activation='relu', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
	keras.layers.Dense(10),
	keras.layers.Activation('softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(reshaped_train_images, mnist.train.labels, batch_size=50, epochs=30)

weights_mat = model.get_weights()[0]
weights = np.split(weights_mat, 25, 3)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(weights[i].reshape(12, 12), cmap=plt.cm.binary)
plt.show()

'''
from keras.callbacks import LambdaCallback

print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics['accuracy'])
model.fit(X_train, 
          y_train, 
          batch_size=batch_size, 
          nb_epoch=5 validation_data=(X_test, y_test), 
          callbacks = [print_weights])
'''