import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print("\n\n\n\n\n")

Ntrain = mnist.train.images.shape[0]
Ntest  = mnist.test.images.shape[0]

reshaped_train_images = np.zeros(shape=(Ntrain, 28, 28, 1))
reshaped_test_images  = np.zeros(shape=(Ntest, 28, 28, 1))
for i in range(Ntrain):
	reshaped_train_images[i] = mnist.train.images[i].reshape(28, 28, 1)
for i in range(Ntest):
  reshaped_test_images[i] = mnist.test.images[i].reshape(28, 28, 1)

# Show train dataset
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(reshaped_train_images[i].reshape(28, 28), cmap='Greys_r')
plt.show()

# Build model
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

# Train model
model.fit(reshaped_train_images, mnist.train.labels, batch_size=50, epochs=2)

# Evaluate and print accuracy
loss, acc = model.evaluate(reshaped_test_images, mnist.test.labels)
print('Accuracy:', acc)


# Find and show set of first 25 trained weights
weights_mat = model.get_weights()[0]
weights = np.split(weights_mat, 25, 3)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(weights[i].reshape(12, 12), cmap='Greys_r')
plt.show()

# Find and show 12 best activations for first 5 filters
intermediate_layer_model = keras.Model(inputs=model.inputs, outputs=model.layers[0].output)
intermediate_outputs = intermediate_layer_model.predict(reshaped_test_images)
transp_outputs = np.transpose(intermediate_outputs, [0, 3, 1, 2]) #NCHW

best_act_values = [0.0 for _ in range(12)]
best_patches = [np.zeros(shape=(12, 12)) for _ in range(12)]

for i in range(len(transp_outputs)): # for all images in test dataset
  activations = transp_outputs[i]

  for j in range(5): # first 5 filters
    activation = activations[j]

    for (x,y), val in np.ndenumerate(activation):
      for k in range(len(best_act_values)):
        if val > best_act_values[k]:
          best_act_values[k] = val
          best_patches[k] = reshaped_test_images[i, (x*2):(x*2+12), (y*2):(y*2+12)]
          break

plt.figure(figsize=(10,10))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(best_patches[i].reshape(12, 12), cmap='Greys_r')
plt.show()
