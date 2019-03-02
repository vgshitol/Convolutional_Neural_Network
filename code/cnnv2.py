import numpy as np

# Used for Loading MNIST
from struct import unpack

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam

import matplotlib.pyplot as plt

DATA_PATH = '../data/'
IMAGE_FILE_TRAIN = 'train-images-idx3-ubyte'
LABEL_FILE_TRAIN = 'train-labels-idx1-ubyte'
IMAGE_FILE_TEST = 't10k-images-idx3-ubyte'
LABEL_FILE_TEST = 't10k-labels-idx1-ubyte'

# Default Parameters
ACTIVATION = None  # BEST : tanh
KERNEL_SIZE = (3, 3)
NUM_FILTER = [8, 16]
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# Set this parameter to True for tuning grid search
TUNING = False
ACTIVATIONS = ['relu', 'tanh', 'sigmoid']

imageTest = []
labelTest = []
imageTrain = []
labelTrain = []


def readMNIST(imagefile, labelfile):
    img = open(imagefile, 'rb')
    lbl = open(labelfile, 'rb')

    img.read(4)
    img_nums = img.read(4)
    img_rows = img.read(4)
    img_cols = img.read(4)

    img_nums = unpack('>I', img_nums)[0]
    img_rows = unpack('>I', img_rows)[0]
    img_cols = unpack('>I', img_cols)[0]
    # print(img_nums, img_rows, img_cols)
    img_count = img_nums * img_rows * img_cols

    buffer = img.read(img_count)
    imageSet = np.frombuffer(buffer, dtype='uint8')
    imageSet = imageSet.reshape(img_nums, img_rows, img_cols, 1)
    # print(imageSet.shape)

    lbl.read(4)
    lbl_nums = lbl.read(4)
    lbl_nums = unpack('>I', lbl_nums)[0]
    # print(lbl_nums)
    buffer = lbl.read(lbl_nums)
    labelSet = np.frombuffer(buffer, dtype='uint8')
    # print(labelSet.shape)
    print(imageSet.shape)
    print(labelSet.shape)
    return (imageSet, labelSet)


def plot_acc(x, acc, xlbl='Batch'):
    # Accuracy
    plt.figure(1)
    plt.plot(x, acc, color='red', label='Accuracy')
    plt.xlabel(str(xlbl))
    plt.ylabel('Accuracy')
    plt.legend('acc')
    plt.title('Accuracy Vs ' + str(xlbl))
    # plt.savefig(DATA_PATH + xlabel + '_acc.png')
    plt.show()


def plot_accuracy(acc_train, acc_test, title, figname="Acc_plot.png"):
    plt.plot(acc_train, color='red', label='Training Accuracy')
    plt.plot(acc_test, color='blue', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title(title)
    plt.savefig(figname)
    plt.show()
    return plt


def plot_loss(loss_train, loss_test, title, figname="Loss_plot.png"):
    plt.plot(loss_train, color='red', label='Training Loss')
    plt.plot(loss_test, color='blue', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title(title)
    plt.savefig(figname)
    plt.show()
    return plt


def DNN_model(activation=None, kernel_size=None, learning_rate=None):
    if kernel_size is None:
        kernel_size = KERNEL_SIZE

    if activation is None:
        activation = ACTIVATION

    if learning_rate is None:
        learning_rate = LEARNING_RATE

    model = Sequential()
    model.add(Conv2D(NUM_FILTER[0], kernel_size=kernel_size, activation=activation, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(NUM_FILTER[1], kernel_size=kernel_size, activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(20, activation=activation))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', 'mae'])
    print(model.summary())
    return model


def tune_Hybrid(X_train, y_train):
    grid_param = dict(epochs=[5, 10, 15, 20],
                      batch_size=[64, 128, 256, 512],
                      kernel_size=[(2, 2), (3, 3), (4, 4)],
                      learning_rate=[1e-4, 5e-4, 1e-3, 2e-3])
    print(grid_param)
    model = KerasClassifier(build_fn=DNN_model)
    gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, cv=2, n_jobs=2)

    result = gd_sr.fit(X_train, y_train)

    print("Best: %f using %s" % (gd_sr.best_score_, gd_sr.best_params_))
    means = gd_sr.cv_results_['mean_test_score']
    stds = gd_sr.cv_results_['std_test_score']
    params = gd_sr.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return model, result


def decide_activation(X_train, y_train):
    bestparam = []
    for activation in ACTIVATIONS:
        model = DNN_model(activation=activation, kernel_size=KERNEL_SIZE, learning_rate=LEARNING_RATE)
        history = model.fit(X_train, y_train, validation_split=0.20, epochs=EPOCHS, batch_size=BATCH_SIZE)
        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        acc_train = history.history['acc']
        acc_val = history.history['val_acc']
        plot_loss(loss_train, loss_val, title='Epochs Vs Loss(Train, Validation)',
                  figname=DATA_PATH + "activation_" + str(activation) + "loss.png")
        plot_accuracy(acc_train, acc_val, title='Epochs Vs Accuracy(Train, Validation)',
                      figname=DATA_PATH + "activation_" + str(activation) + "acc.png")
        bestparam.append(max(acc_val))

    best_params = ACTIVATION[bestparam.index(max(bestparam))]
    return best_params


def one_hot_encode(data):
    """
    :param data: data to be one hot encoded
    :return: np array with one hot encoding
    """
    print(data.max())
    labels = np.zeros((data.size, data.max() + 1))
    labels[np.arange(data.size), data] = 1
    return labels


def main():
    imageTrain, labelTrain = readMNIST(DATA_PATH + IMAGE_FILE_TRAIN, DATA_PATH + LABEL_FILE_TRAIN)
    imageTest, labelTest = readMNIST(DATA_PATH + IMAGE_FILE_TEST, DATA_PATH + LABEL_FILE_TEST)

    labelTrain = to_categorical(labelTrain)
    labelTest = to_categorical(labelTest)

    tuned_kernel_size, tuned_epochs, tuned_batch_size, tuned_learning_rate, tuned_activation = \
        KERNEL_SIZE, EPOCHS, BATCH_SIZE, LEARNING_RATE, ACTIVATION

    if TUNING:
        model, result = tune_Hybrid(imageTrain, labelTrain)
        print('\n'.join('{}: {}'.format(*k) for k in enumerate(result.cv_results_['params'])))
        tuned_kernel_size = result.best_params_['kernel_size']
        tuned_epochs = result.best_params_['epochs']
        tuned_batch_size = result.best_params_['batch_size']
        tuned_learning_rate = result.best_params_['learning_rate']
        print(result.best_params_)
        tuned_activation = decide_activation(imageTrain, labelTrain)

    model = DNN_model(activation=tuned_activation, kernel_size=tuned_kernel_size, learning_rate=tuned_learning_rate)
    model.fit(imageTrain, labelTrain, validation_split=0.2, epochs=tuned_epochs, batch_size=tuned_batch_size)
    _train_results = model.evaluate(imageTrain, labelTrain)
    _test_results = model.evaluate(imageTest, labelTest)
    print(_test_results, _train_results)
    preds = model.predict(imageTest)
    cnn_one_hot = one_hot_encode(preds.argmax(axis=1))
    np.savetxt(DATA_PATH + "mnist.csv", cnn_one_hot, delimiter=",", fmt='%i')


if __name__ == '__main__':
    main()
