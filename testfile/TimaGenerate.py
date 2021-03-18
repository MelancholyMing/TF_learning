import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")

genner = datagen.flow_from_directory(r"./data",
                                     batch_size=2,
                                     shuffle=False,
                                     save_to_dir=r"./data_gen",
                                     save_prefix='trains_',
                                     save_format="jpg"
                                     )


