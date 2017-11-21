import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def try_mnist(epochs, batch_size):
    # MNISTデータを読込む
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # MNISTデータを加工する
    x_train  = x_train.reshape(60000, 784)
    x_test   = x_test.reshape(10000, 784)
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255
    y_train  = keras.utils.to_categorical(y_train, 10)
    y_test   = keras.utils.to_categorical(y_test, 10)

    # モデルの構築
    model = Sequential()
    model.add(InputLayer(input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # 学習
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    # 検証
    score = model.evaluate(x_test, y_test, verbose=1)
    print()
    print('epochs', epochs)
    print('batch_size', batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score

# 計測
epochs = 30
batch_size = 128
result = [try_mnist(epochs=i, batch_size=batch_size)[1] for i in range(epochs)]

# 可視化
plt.figure()
plt.plot(range(epochs), result)
plt.savefig('image.png')
