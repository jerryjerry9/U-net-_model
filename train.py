import os
import numpy as np
import tensorflow as tf
from PIL import Image
from data_generator import DataGenerator
from model import unet


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    root_path = 'training/img/'
    IDs = [name.rstrip('.pkl')
           for name in os.listdir(root_path)]

    # Parameters
    params = {'dim': (512, 512),
              'batch_size': 1,
              'n_channels': 3,
              'shuffle': True,
              'train': True}

    # Datasets
    partition = {'train': IDs[:int(len(IDs)*0.8)],
                 'validation': IDs[int(len(IDs)*0.8):]}

    # tf.data.Dataset
    batch_size = 2
    AUTOTUNE = tf.data.AUTOTUNE

    # Design model
    model = unet(channels=3)
    train_generator = DataGenerator(partition['train'], **params)
    val_generator = DataGenerator(partition['validation'], **params)

    # callback
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/test/',
                                                 update_freq=1)
    checkpoint_filepath = 'models/test/Unet_{epoch:03d}-{loss:.4f}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=checkpoint_filepath,
                                            save_weights_only=True,
                                            monitor='loss',
                                            mode='min',
                                            save_best_only=True)

    # Train model on dataset
    model.fit(train_generator,
              epochs=1000,
              validation_data=val_generator,
              callbacks=[model_checkpoint_callback, tensorboard])
