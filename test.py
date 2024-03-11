import os 
import pickle
import numpy as np
import tensorflow as tf
#from PIL import Image
from skimage.transform import resize
from data_generator import DataGenerator
from model import unet
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    root_path = '/home/your_ID/testing/test/'
    IDs = [name.rstrip('.pkl')
           for name in os.listdir(root_path)]

    # Parameters
    params = {'dim': (512, 512),
              'batch_size': 1,
              'n_channels': 3,
              'shuffle': False,
              'train': False,
              'predict_dir': root_path}

    # Datasets
    partition = {'test': IDs}

    # tf.data.Dataset
    batch_size = 16
   # AUTOTUNE = tf.data.AUTOTUNE

    # models
    model = unet(channels=3)
    ### correct kernel 888
    ### ori enh model 870
    model.load_weights('models/Unet_921-0.0012.hdf5')
    test_generator = DataGenerator(partition['test'], **params)

    for i, (data, mask) in enumerate(test_generator):
        name = partition['test'][i]
        predict = model.predict(data).squeeze()
        predict = resize(predict, (1000, 1000))
        print(predict.shape)
        with open(f'/home/your_ID/predict/{name}.pkl', 'wb') as f:
            pickle.dump(predict, f)
