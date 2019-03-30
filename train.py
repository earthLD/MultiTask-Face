import numpy as np
import os
import threading
from PIL import Image
from keras.applications.mobilenet import preprocess_input
import tensorflow as tf
import os
from keras import backend as K
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau,CSVLogger, ModelCheckpoint
import keras.optimizers as optimizers
import build_model
from load_data import load_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config =  tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
K.set_session(tf.Session(config=config))



def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < 0.5, 0.5 * x ** 2, x - 0.5)
    return  K.mean(x)

def loss_distance(label_true, label_pred):
    y_true = label_true[::2]
    x_true = label_true[1::2]
    
    y_pred = label_pred[::2]
    x_pred = label_pred[1::2]
    
    x_abs = x_true - x_pred
    y_abs = y_true - y_pred
    
    return K.mean(K.sqrt(K.square(x_abs) + K.square(y_abs)))

def Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2):
    """
    focal loss for multi-class classification
    fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
    :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
    :param y_pred: prediction after softmax shape of [batch_size, nb_class]
    :param alpha:
    :param gamma:
    :return:
    """
    # # parameters
    y_pred += K.epsilon()
    ce =-y_true * K.log(y_pred)
    weight = K.pow(1 - y_pred, gamma) * y_true
    fl = ce * weight * alpha
    reduce_fl = tf.reduce_sum(fl, axis=-1)

    return reduce_fl

def train():
    train_generator, test_generator = load_data()
    model = build_model.build_model_4()
    # model.load_weights('model/model_small1_TA_V4_1.h5')
    checkpoint = ModelCheckpoint(filepath='./model/model_small1_TA_V4_1.h5', verbose=1, 
                save_best_only=True, save_weights_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=0.00001)

    model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  loss={'pred_age': smoothL1, 
                        'pred_kp': loss_distance, 
                        'pred_g':'categorical_crossentropy', 
                        'pred_emo': 'categorical_crossentropy', 
                        'pred_eth': 'categorical_crossentropy'}, 
                  loss_weights= [1,1,1,1,1]
    )


    model.fit_generator(
                generator=train_generator,
                steps_per_epoch= 1000,
                epochs=100,
                validation_data=test_generator,
                validation_steps=1000,
                workers=0,
                use_multiprocessing=True,
                callbacks=[checkpoint, reduce_lr]
    )
    
if __name__ == '__main__':
    train()