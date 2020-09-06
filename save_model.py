import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import datasets

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()


inputs = layers.Input((28, 28, 1), name='input_layer')

# Feature Extraction
net = layers.Conv2D(32, (3, 3), padding='SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D((2, 2))(net)
net = layers.Dropout(0.25)(net)

net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, (3, 3), padding='SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D((2, 2))(net)
net = layers.Dropout(0.25)(net)  # (None, 7, 7, 64)

# Classification
net = layers.Flatten()(net)  # (None, 3136=7*7*64)  
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(10)(net)  # num_classes
net = layers.Activation('softmax')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='Basic_CNN')

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.Accuracy()]
              )

train_x = train_x[..., tf.newaxis]
train_y = tf.keras.utils.to_categorical(train_y, 10)

test_x = test_x[..., tf.newaxis]
test_y = tf.keras.utils.to_categorical(test_y, 10)

model.fit(train_x, train_y,
          validation_data=(test_x, test_y),
          batch_size=32, 
          shuffle=True,
          epochs=1)


# Save Model

save_path = 'checkpoints'

checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, #저장 경로
                                                monitor = 'accuracy', #무엇을 볼건지
                                                save_best_only = True, # 가장 좋을 때만 저장
                                                #false면 매번 저장
                                                mode='max') # 모니터링 대상이 max인지 min인지

logdir = 'log'
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(train_x, train_y,
          validation_data=(test_x, test_y),
          batch_size=32, 
          shuffle=True,
          epochs=2,
          callbacks=[checkpoint, tensorboard])

model.save("cnn_basic.h5")


# Load Model

new_model = tf.keras.models.load_model('cnn_basic.h5')

new_model.summary()

