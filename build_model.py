from keras.layers import Input, SeparableConv2D, GlobalMaxPooling2D ,Conv2D, GlobalAveragePooling2D, MaxPooling2D, Concatenate, Activation
from keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization, AveragePooling2D, multiply, Flatten, UpSampling2D
import tensorflow as tf
import numpy as np


def conv_bn_act(inputs, n_filters=64, kernel=(2, 2), strides=1, activation='relu'):

    conv = SeparableConv2D(n_filters, kernel_size= kernel, strides = strides, data_format='channels_last')(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)

    return conv

def ARM(inputs, n_filters):
    
    # ARM (Attention Refinement Module)
    # Refines features at each stage of the Context path
    # Negligible computation cost
    arm = AveragePooling2D(pool_size=(1, 1), padding='same', 
                           data_format='channels_last')(inputs)
    arm = conv_bn_act(arm, n_filters, (1, 1), activation='sigmoid')
    arm = multiply([inputs, arm])

    return arm

def Sepconv(inputs, n_filters, stride=2):
    net = SeparableConv2D(filters=n_filters, kernel_size=3, strides=stride, padding='same')(inputs) # 1/2
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # if stride == 2:
    #     net = MaxPooling2D((2, 2))(net)
    return net


def build_model_2():
    input_img = Input(shape=[192, 192, 3])
    net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_img) # 1/2
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net = SeparableConv2D(filters=64, kernel_size=3, strides=1, padding='same')(pool)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net = SeparableConv2D(filters=128, kernel_size=3, strides=1, padding='same')(pool) # 1/4
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net = SeparableConv2D(filters=256, kernel_size=3, strides=1, padding='same')(pool)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net1 = pool
    net = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same')(pool) # 1/8
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net2 = pool
    net = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same')(pool) # 1/8
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same')(pool) # 1/8
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same')(pool) # 1/8
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net3 = net
    net1 = MaxPooling2D((2,2))(net1)
    conv_all = Concatenate()([net1,net2,net3])
    
    pool_age = ARM(conv_all,1280)
    net_age = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_age) # 1/2
    net_age = GlobalAveragePooling2D()(net_age)
    net_age = Dropout(0.4)(net_age)
    net_age = Dense(256, activation='relu')(net_age)
    net_age = Dropout(0.4)(net_age)
    net_age = Dense(1, name='pred_age')(net_age)
#     print('net:',net.shape)

    pool_kp = ARM(conv_all,1280)
    net_kp = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_kp) # 1/2
    net_kp = GlobalAveragePooling2D()(net_kp)
    net_kp = Dropout(0.4)(net_kp)
    net_kp = Dense(256, activation='relu')(net_kp)
    net_kp = Dropout(0.4)(net_kp)
    net_kp = Dense(10, activation='sigmoid', name='pred_kp')(net_kp)
    
    pool_g = ARM(conv_all,1280)
    net_g = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_g) # 1/2
    net_g = GlobalAveragePooling2D()(net_g)
    net_g = Dropout(0.4)(net_g)
    net_g = Dense(256, activation='relu')(net_g)
    net_g = Dropout(0.4)(net_g)
    net_g = Dense(2, activation='softmax', name='pred_g')(net_g)
    
    pool_emo = ARM(conv_all,1280)
    net_emo = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_emo) # 1/2
    net_emo = GlobalAveragePooling2D()(net_emo)
    net_emo = Dropout(0.4)(net_emo)
    net_emo = Dense(256, activation='relu')(net_emo)
    net_emo = Dropout(0.4)(net_g)
    net_emo = Dense(3, activation='softmax', name='pred_emo')(net_emo)
    
    pool_eth = ARM(conv_all,1280)
    net_eth = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_eth) # 1/2
    net_eth = GlobalAveragePooling2D()(net_eth)
    net_eth = Dropout(0.4)(net_eth)
    net_eth = Dense(256, activation='relu')(net_eth)
    net_eth = Dropout(0.4)(net_g)
    net_eth = Dense(4, activation='softmax', name='pred_eth')(net_eth)
    
    model = Model(inputs = input_img, outputs=[net_age, net_kp, net_g, net_emo, net_eth])
    model.summary()
    return model

def build_model_3():
    input_img = Input(shape=[192, 192, 3])
    net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_img) # 1/2
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net = SeparableConv2D(filters=64, kernel_size=3, strides=1, padding='same')(pool)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net = SeparableConv2D(filters=128, kernel_size=3, strides=1, padding='same')(pool) # 1/4
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net1 = pool
    net = SeparableConv2D(filters=256, kernel_size=3, strides=1, padding='same')(pool)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net2 = pool
    net = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same')(pool) # 1/8
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    pool = MaxPooling2D((2,2))(net)
    net = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same')(pool) # 1/8
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same')(pool) # 1/8
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = SeparableConv2D(filters=512, kernel_size=3, strides=1, padding='same')(pool) # 1/8
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net3 = net
    net1 = MaxPooling2D((4,4))(net1)
    net2 = MaxPooling2D((2,2))(net2)
    conv_all = Concatenate()([net1,net2,net3])
    
    pool_age = ARM(conv_all,896)
    net_age = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_age) # 1/2
    net_age = GlobalAveragePooling2D()(net_age)
    net_age = Dropout(0.4)(net_age)
    net_age = Dense(256, activation='relu')(net_age)
    net_age = Dropout(0.4)(net_age)
    net_age = Dense(1, name='pred_age')(net_age)
#     print('net:',net.shape)

    pool_kp = ARM(conv_all,896)
    net_kp = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_kp) # 1/2
    net_kp = GlobalAveragePooling2D()(net_kp)
    net_kp = Dropout(0.4)(net_kp)
    net_kp = Dense(256, activation='relu')(net_kp)
    net_kp = Dropout(0.4)(net_kp)
    net_kp = Dense(10, activation='sigmoid', name='pred_kp')(net_kp)
    
    pool_g = ARM(conv_all,896)
    net_g = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_g) # 1/2
    net_g = GlobalAveragePooling2D()(net_g)
    net_g = Dropout(0.4)(net_g)
    net_g = Dense(256, activation='relu')(net_g)
    net_g = Dropout(0.4)(net_g)
    net_g = Dense(2, activation='softmax', name='pred_g')(net_g)
    
    pool_emo = ARM(conv_all,896)
    net_emo = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_emo) # 1/2
    net_emo = GlobalAveragePooling2D()(net_emo)
    net_emo = Dropout(0.4)(net_emo)
    net_emo = Dense(256, activation='relu')(net_emo)
    net_emo = Dropout(0.4)(net_g)
    net_emo = Dense(3, activation='softmax', name='pred_emo')(net_emo)
    
    pool_eth = ARM(conv_all,896)
    net_eth = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool_eth) # 1/2
    net_eth = GlobalAveragePooling2D()(net_eth)
    net_eth = Dropout(0.4)(net_eth)
    net_eth = Dense(256, activation='relu')(net_eth)
    net_eth = Dropout(0.4)(net_g)
    net_eth = Dense(4, activation='softmax', name='pred_eth')(net_eth)
    
    model = Model(inputs = input_img, outputs=[net_age, net_kp, net_g, net_emo, net_eth])
    model.summary()
    return model

def build_model_4():
    input_img = Input(shape=[192, 192, 3])
    net = Sepconv(input_img, 32)
    net = Sepconv(net, 64)
    net = Sepconv(net, 128)
    net1 = net
    net = Sepconv(net, 256)
    net2 = net
    net = Sepconv(net, 512)
    net = Sepconv(net, 512, stride = 1)
    net = Sepconv(net, 512, stride = 1)
    net = Sepconv(net, 512, stride = 1)
    net3 = net
    # net3 = UpSampling2D(size=(2, 2))(net)
    net1 = MaxPooling2D((4, 4))(net1)
    net2 = MaxPooling2D((2, 2))(net2)
    conv_all = Concatenate()([net1,net2,net3])
    
    
    pool_age = ARM(conv_all,896)
    net_age = Sepconv(pool_age, 256)
    net_age = GlobalAveragePooling2D()(net_age)
    net_age = Dropout(0.4)(net_age)
    # net_age = Dense(256, activation='relu')(net_age)
    # net_age = Dropout(0.4)(net_age)
    net_age = Dense(1, name='pred_age')(net_age)
#     print('net:',net.shape)

    pool_kp = ARM(conv_all,896)
    net_kp = Sepconv(pool_kp, 256)
    net_kp = GlobalAveragePooling2D()(net_kp)
    net_kp = Dropout(0.4)(net_kp)
    # net_kp = Dense(256, activation='relu')(net_kp)
    # net_kp = Dropout(0.4)(net_kp)
    net_kp = Dense(10, activation='sigmoid', name='pred_kp')(net_kp)
    
    pool_g = ARM(conv_all,896)
    net_g = Sepconv(pool_g, 256)
    net_g = GlobalAveragePooling2D()(net_g)
    net_g = Dropout(0.4)(net_g)
    # net_g = Dense(256, activation='relu')(net_g)
    # net_g = Dropout(0.4)(net_g)
    net_g = Dense(2, activation='softmax', name='pred_g')(net_g)
    
    pool_emo = ARM(conv_all,896)
    net_emo = Sepconv(pool_emo, 256)
    net_emo = GlobalAveragePooling2D()(net_emo)
    net_emo = Dropout(0.4)(net_emo)
    # net_emo = Dense(256, activation='relu')(net_emo)
    # net_emo = Dropout(0.4)(net_g)
    net_emo = Dense(3, activation='softmax', name='pred_emo')(net_emo)
    
    pool_eth = ARM(conv_all,896)
    net_eth = Sepconv(pool_eth, 256)
    net_eth = GlobalAveragePooling2D()(net_eth)
    net_eth = Dropout(0.4)(net_eth)
    # net_eth = Dense(256, activation='relu')(net_eth)
    # net_eth = Dropout(0.4)(net_g)
    net_eth = Dense(4, activation='softmax', name='pred_eth')(net_eth)
    
    model = Model(inputs = input_img, outputs=[net_age, net_kp, net_g, net_emo, net_eth])
    model.summary()
    return model

def build_model_5():
    input_img = Input(shape=[192, 192, 3])
    net = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input_img) # 1/2
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    
    net = Sepconv(input_img, 64, 1)
    net = Sepconv(net, 128, 2)
    net = Sepconv(net, 128, 1)
    net1 = net
    net = Sepconv(net, 256, 2)
    net = Sepconv(net, 256, 1)
    net2 = net
    net = Sepconv(net, 512, 2)
    net = Sepconv(net, 512, 1)
    net = Sepconv(net, 512, 1)
    net = Sepconv(net, 512, 1)
    
    net1 = MaxPooling2D((4, 4))(net1)
    net2 = MaxPooling2D((2, 2))(net2)
    conv_all = Concatenate()([net1,net2,net3])
    
    
    net_age = ARM(conv_all, 896)
    net_age = Sepconv(net_age, 128, 2)
    net_age = Sepconv(net_age, 256, 2)
    net_age = GlobalAveragePooling2D()(net_age)
    net_age = Dropout(0.4)(net_age)
    net_age = Dense(1, name='pred_age')(net_age)
    
    net_g = ARM(conv_all, 896)
    net_g = Sepconv(net_g, 128, 2)
    net_g = Sepconv(net_g, 256, 2)
    net_g = GlobalAveragePooling2D()(net_g)
    net_g = Dropout(0.4)(net_g)
    net_g = Dense(2, activation='softmax', name='pred_g')(net_g)
    
    
    net_kp = ARM(conv_all, 896)
    net_kp = Sepconv(net_kp, 128, 2)
    net_kp = Sepconv(net_kp, 256, 2)
    net_kp = GlobalAveragePooling2D()(net_kp)
    net_kp = Dropout(0.4)(net_kp)
    net_kp = Dense(10, activation='sigmoid', name='pred_kp')(net_kp)
    
    net_emo = ARM(conv_all, 896)
    net_emo = Sepconv(net_emo, 128, 2)
    net_emo = Sepconv(net_emo, 256, 2)
    net_emo = GlobalAveragePooling2D()(net_emo)
    net_emo = Dropout(0.4)(net_emo)
    net_emo = Dense(3, activation='softmax', name='pred_emo')(net_emo)
    
    net_eth = ARM(conv_all, 896)
    net_eth = Sepconv(net_eth, 128, 2)
    net_eth = Sepconv(net_eth, 256, 2)
    net_eth = GlobalAveragePooling2D()(net_eth)
    net_eth = Dropout(0.4)(net_eth)
    net_eth = Dense(4, activation='softmax', name='pred_eth')(net_eth)
    
    model = Model(inputs = input_img, outputs=[net_age, net_kp, net_g, net_emo, net_eth])
    model.summary()
    return model