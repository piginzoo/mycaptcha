#-*- coding:utf-8 -*-  
#__author__ = 'piginzoo'
#__date__ = '2018/2/1'
from __future__ import print_function
import cv2
import h5py
import codecs
import numpy as np
import os

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import TensorBoard  
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import input_image,input_label,cnn
import logging as logger
import crash_debug

'''
    实现两个网络，分别实现识别字符数，识别字符。

    1.该如何训练呢？
      1.1 该把数据准备好呢？准备成啥样啥格式呢？
      1.2 批次是不是不用考虑？一次加载进去，然后tf/keras自己靠batch参数控制？keras不是有批次输入么？
    2.该如何中途保存呢？
    3.该如何输出呢？
    4.如何验证正确率和loss？loss是错误率么？是一个batch评价一次么？
    其他问题：
      - 去研究一下minst的数据加载，如何填充(？, 1, 36, 100)第一个维度的？
          应该是一次都加载进去么？

'''

if __name__ == '__main__':

    # 设置默认的level为DEBUG
    # 设置log的格式
    logger.basicConfig(
        level=logger.INFO,
        format="[%(levelname)s] %(message)s"
    )


    letters = list('0123456789abcdefghijklmnopqrstuvwxyz')
    weight_decay = 0.001

    image_dir = "image/type5_train/images/"
    label_path = "image/type5_train/type5_train.csv"

    # batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。所以要适当选择
    batch_size = 50
    # 完整迭代次数
    epochs = 5

    #3:表示就3中结果4，5，6个字符    
    num_classes = 3

    model_name = 'model/num_model.h5'

    # 训练集
    x_train = input_image.load_all_image_by_dir(image_dir)
    y_train = input_label.load_label(label_path)
    # 把4,5,6变成 one-hot verctor
    y_train = keras.utils.np_utils.to_categorical(y_train - 4, num_classes)

    if os.path.exists(model_name):
        num_model = load_model(model_name)
        logger.info("加载已经存在的训练模型%s",model_name)
    else:
        logger.info("创建识别字符长度的模型")
        input_shape = None 
        if K.image_data_format() == 'channels_first':
            input_shape = (1,36,100)
        else:
            input_shape = (36,100,1)   #这个shape不包含第一个维度，也就是图片数量

        
        #定义寻找字符个数的CNN
        num_model = cnn.create_model(input_shape,num_classes) 

        #定义判断字符的CNN
        #char_model = cnn.create_model()


    #训练期间保存checkpoint，防止整个crash
    #checkpoint_path = "num-model-{epoch:02d}-{val_acc:.2f}.hdf5" #这个是对save_best_only=False的文件名
    checkpoint_path = "num_model_checkout.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # 令人兴奋的训练过程
    history = num_model.fit(
        x_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=epochs,
        verbose=1, 
        callbacks=[TensorBoard(log_dir='./log'),checkpoint],
        validation_split=0.1)#拿出10%来不参与训练，而用做中途的验证

    logger.info("训练的过程：%r",history)

    num_model.save(model_name)
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])