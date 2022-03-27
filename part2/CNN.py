import numpy as np, matplotlib.pyplot as plt
from os.path import join
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, MaxPooling2D, BatchNormalization, Activation, \
    Conv2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Nadam, Adam
import seaborn as sbn
from tensorflow.keras.models import load_model



def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}

def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'Yes TFL'}):
    print(images.shape, ' ', labels.shape)
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,
                      sharex=True, sharey=True)[1]  # .flatten()
    idxs = np.random.randint(0, images.shape[0], n)
    for i, idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None: title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)

def tfl_model():
    input_shape = (81, 81, 3)

    model = Sequential()

    def conv_bn_relu(filters, **conv_kw):
        model.add(Conv2D(filters, use_bias=False, kernel_initializer='he_normal', **conv_kw))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def dense_bn_relu(units):
        model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def spatial_layer(count, filters):
        for i in range(count):
            conv_bn_relu(filters, kernel_size=(3, 3))
        conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

    conv_bn_relu(32, kernel_size=(3, 3), input_shape=input_shape)
    spatial_layer(1, 32)
    spatial_layer(2, 64)
    spatial_layer(2, 96)

    model.add(Flatten())
    dense_bn_relu(96)
    model.add(Dense(2, activation='softmax'))
    return model


def validate_data():
    data_dir = '/content/gdrive/MyDrive/miris_data'
    datasets = {
        'val': load_tfl_data(join(data_dir, 'val')),
        'train': load_tfl_data(join(data_dir, 'train')),
    }
    for k, v in datasets.items():
        print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape, np.mean(v['labels'] == 1) * 100))

    viz_my_data(num=(6, 6), **datasets['train'])


def build_and_train_CNN():
    model = tfl_model()
    model.summary()
    data_dir = '/content/gdrive/MyDrive/miris_data'
    datasets = {
        'val': load_tfl_data(join(data_dir, 'val')),
        'train': load_tfl_data(join(data_dir, 'train')),
    }
    # prepare our model
    model = tfl_model()
    model.compile(optimizer=Nadam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
    train, val = datasets['train'], datasets['val']
    # train it, the model uses the 'train' dataset for learning. We evaluate the "goodness" of the model, by predicting the label of the images in the val dataset.
    model.fit(train['images'], train['labels'], validation_data=(val['images'], val['labels']), epochs=10)

    model.save("model.h5")
    return model


