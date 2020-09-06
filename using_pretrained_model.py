import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers


# transfer learning using pretrained model

(train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()

base_model = tf.keras.applications.VGG16(include_top = False,
                                    weights = 'imagenet',
                                    input_shape = (32,32,3),
                                    classes = 10)

#base_model.summary()


base_model.trainable = False


num_classes = 10 # Hyperparameter


gap = layers.GlobalAveragePooling2D()
pred = layers.Dense(num_classes, activation="softmax")


model = tf.keras.Sequential([base_model, gap, pred]) 


#model.summary()


model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer = tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
train_x.shape , train_y.shape

train_x = train_x / 255.
train_y = tf.keras.utils.to_categorical(train_y, 10)

test_x = test_x / 255.
test_y = tf.keras.utils.to_categorical(test_y, 10)


hist = model.fit(train_x, train_y,
                 validation_data=(test_x, test_y),
                 epochs = 10)