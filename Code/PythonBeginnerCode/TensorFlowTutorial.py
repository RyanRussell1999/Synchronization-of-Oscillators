import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Defining NN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # Input Layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Hidden Layer 1
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Hidden Layer 2
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Output Layer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([x_test])

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()
