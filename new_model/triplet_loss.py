"""Define functions to create the triplet loss with online triplet mining."""
# https://github.com/phongdinhv/triplet-loss-keras-mnist/blob/master/triplet_loss.ipynb

import tensorflow as tf
import numpy as np
from keras import backend as K

def triplets_loss(y_true, y_pred):
    embeddings = y_pred
    anchor_positive = embeddings[:10]
    negative = embeddings[10:]
    
    # Compute pairwise distance between all of anchor-positive
    dot_product = K.dot(anchor_positive, K.transpose(anchor_positive))
    square = K.square(anchor_positive)
    a_p_distance = K.reshape(K.sum(square, axis=1), (-1,1)) - 2.*dot_product  + K.sum(K.transpose(square), axis=0) + 1e-6
    a_p_distance = K.maximum(a_p_distance, 0.0) ## Numerical stability

    # Compute distance between anchor and negative
    dot_product_2 = K.dot(anchor_positive, K.transpose(negative))
    negative_square = K.square(negative)
    a_n_distance = K.reshape(K.sum(square, axis=1), (-1,1)) - 2.*dot_product_2  + K.sum(K.transpose(negative_square), axis=0)  + 1e-6
    a_n_distance = K.maximum(a_n_distance, 0.0) ## Numerical stability
    
    hard_negative = K.reshape(K.min(a_n_distance, axis=1), (-1, 1))
    
    distance = (a_p_distance - hard_negative + 0.2)
    loss = K.mean(K.maximum(distance, 0.0))/(2.)
            
    return loss
