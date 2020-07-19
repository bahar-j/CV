import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, datasets

import numpy as np


# Hyperparameter Tuning

num_classes = 100  
input_shape = (32, 32, 3)


# Prepare Dataset

(train_x, train_y), (test_x, test_y) = datasets.cifar100.load_data()

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

train_y = tf.keras.utils.to_categorical(train_y, 100)
test_y = tf.keras.utils.to_categorical(test_y, 100)

train_x = train_x / 255
test_x = test_x / 255

print(train_y.shape, test_y.shape)


# Build Model

inputs = layers.Input(input_shape)

# Feature Extraction
net = layers.Conv2D(32, 3, strides=(1, 1), padding="SAME")(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, 3, strides=(1, 1), padding="SAME")(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D((2, 2))(net)
net = layers.Dropout(0.5)(net)

net = layers.Conv2D(64, 3, strides=(1, 1), padding="SAME")(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, 3, strides=(1, 1), padding="SAME")(net)
net = layers.Activation('relu')(net)
net = layers.MaxPool2D((2, 2))(net)
net = layers.Dropout(0.5)(net)

# Fully Connected(Classification)
net = layers.Flatten()(net)
net = layers.Dense(512)(net) # 3236개의 노드 -> 512개의 노드 
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(num_classes)(net)
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs = inputs, outputs = net, name='basic_cnn')


# Optimization Setting

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer = tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


# Training Model

hist = model.fit(train_x, train_y,
                 batch_size = 32,
                 shuffle = True,
                 epochs = 20)


# Evaluate

histories = hist.history
print(histories.keys())

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(histories['loss'])
plt.title("Loss Curve")

plt.subplot(122)
plt.plot(histories['accuracy'])
plt.ylim(0, 1)
plt.title('Acccuracy Curve')

plt.show()


logits = model.predict(test_x)

print(logits.shape)

print(np.argmax(logits[0]))
print(np.max(logits[0]))

plt.imshow(test_x[0])
plt.title(np.argmax(logits[0]))
plt.show()

preds = np.argmax(logits, -1)
print(preds.shape)

print(preds[0])

plt.hist(preds)
plt.hist(np.argmax(test_y, -1), color= 'red', alpha = 0.5)
plt.show()
