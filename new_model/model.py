import os
import tensorflow as tf
import keras.models
from keras import backend as K
from keras.layers import Input, Concatenate
from keras.models import Model
    
def get_model(n_frames, n_mels, n_conditions, lr):
    # InceptionV3 is chosen for the base model##############################################
    base_model =  tf.keras.applications.MobileNetV3Large(include_top=False,
                                                   input_shape=(n_frames, n_mels, 3),
                                                   weights=None)
#     base_model.trainable = False
    
    # The input layer have the shape of 3-Dimention (64, 128, 3)############################
    x = Input(shape=(n_frames, n_mels, 1))
    h = x
    h = Concatenate()([h, h, h])
    
    # The final model#######################################################################
    h = base_model(h)
    flatten = tf.keras.layers.Flatten()(h)
    embedding_layer = tf.keras.layers.Dense(units=n_conditions)(flatten)
    model = Model(x, embedding_layer)
    
    # Compile###############################################################################
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

    return model

#########################################################################

def load_model(file_path):
    return keras.models.load_model(file_path, compile=False)

def clear_session():
    K.clear_session()
