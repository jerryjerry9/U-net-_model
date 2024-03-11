import pickle
import numpy as np
from tensorflow import keras
from skimage.transform import resize
from PIL import Image


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, predict_dir=None,
                 batch_size=32, dim=(512, 512),
                 n_channels=1, shuffle=True,
                 train=True):
        'Initialization'
        self.predict_dir = predict_dir
        self.train = train
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            band_data = pickle.load(f)
        return band_data
    def load_label(self, path):
        with open(path, 'rb') as f:
            band_data = pickle.load(f)
        return band_data


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 3))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        if self.train is True:
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                # test as variable in function
                img = self.load_pickle(f'training/img/{ID}.pkl')
                X[i] = resize(img, (512, 512))

                # Store label
                mask = self.load_label(f'training/label/{ID.replace("rgb", "label")}.pkl')[..., np.newaxis]
                y[i] = resize(mask, (512, 512))

        elif self.train is False:
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                # test as variable in function
                img = self.load_pickle(f'{self.predict_dir}{ID}.pkl')
                X[i] = resize(img, (512, 512))
        return X, y
