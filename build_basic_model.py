import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import datasets


#Data Preprocessing
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x = train_x[..., tf.newaxis]
train_y = tf.keras.utils.to_categorical(train_y, 10)

#Modeling

##Feature Extraction
inputs = layers.Input((28, 28, 1), name="input_layer")

net = layers.Conv2D(32, (3,3), padding="SAME")(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding = "SAME")(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D((2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, (3,3), padding="SAME")(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding = "SAME")(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D((2, 2))(net)
net = layers.Dropout(0.25)(net) # (None, 7, 7, 64)


## Classification
net = layers.Flatten()(net) # (None, 3136(노드, 픽셀)) 3136 = 7 * 7 * 64
net = layers.Dense(512)(net) 
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(10)(net) # Number of classes
net = layers.Activation('softmax')(net)


## Model made
model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')


# Optimization Setting
model.compile(loss = tf.keras.losses.categorical_crossentropy, 
              optimizer = tf.keras.optimizers.Adam(), 
              metrics = [tf.keras.metrics.Accuracy()])

# Training
model.fit(train_x, train_y,validation_data=(test_x, test_y), 
         batch_size = 32,
         shuffle = True,
         epochs = 1)