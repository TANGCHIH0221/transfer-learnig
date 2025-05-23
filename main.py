import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train_nor = x_train.astype('float32')/255
x_test_nor = x_test.astype('float32')/255

#獨熱
num_classes=10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#model
#input layer
input_dim = x_train.shape[1:]
inputs = tf.keras.Input(shape=input_dim, name='input_layer')
#first CNN Block
#conv2d:32,3*3 filter,relu 以及增加conv 深度
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),activation='relu',padding='same', name='conv1_1')(inputs)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),activation='relu',padding='same', name='conv1_2')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),name='pool1')(x)

#second
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name= 'conv2_1')(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name= 'conv2_2')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='pool2')(x)

x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='conv3_1')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)

x =tf.keras.layers.Flatten(name='flatten')(x)

#fully connect
x = tf.keras.layers.Dense(units=256, activation='relu', name='des1')(x)
x = tf.keras.layers.Dropout(0.3)(x)
#output
outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax', name='output')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CNN')
model.summary()
#編譯
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train
epochs = 25
batch_size = 128
history = model.fit(x_train_nor, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test_nor, y_test))
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout() # 調整子圖間距
plt.show()

# 6. 評估模型 (在測試集上)
print("\n開始評估模型在測試集上的表現...")
test_loss, test_acc = model.evaluate(x_test_nor, y_test, verbose=2) # verbose=2 會打印每個批次的結果
print(f"\n測試集上的損失 (Test loss): {test_loss}")
print(f"測試集上的準確率 (Test accuracy): {test_acc}")
