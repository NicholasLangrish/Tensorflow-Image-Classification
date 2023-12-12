import tensorflow as tf #import tensorflow
import numpy as np #import numpy
import matplotlib.pyplot as plt #import matplotlib

fashion_mnist=tf.keras.datasets.fashion_mnist #shortcut to dataset
(all_images, all_labels), (test_images, test_labels) = fashion_mnist.load_data() #load data and labels to variables

images = 10000 #set number of training images
train_images=all_images[:images]
train_labels=all_labels[:images]

train_images=train_images/255.0 #scale pixel values from 0-255 to 0-1
test_images=test_images/255.0 #scale pixel values from 0-255 to 0-1

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #reformat images from 28*28 to 784*1
    tf.keras.layers.Dense(2048, activation='relu'),  #number of nodes and activation type
    tf.keras.layers.Dense(10) #number of end values
])

model.compile(optimizer='adam', #how model is updated through training
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #minimise losses in training
              metrics=['accuracy']) #tested metrics
 
model.fit(train_images, train_labels, epochs=10) #train model
test_loss, test_acc = model.evaluate(test_images,  test_labels, workers=2) #test model