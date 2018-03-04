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

from keras.callbacks import ModelCheckpoint

import label,image_process,cnn,debug,crash_debug
import logging as logger


'''
    实现两个网络，分别实现识别字符数，识别字符。

    1.该如何训练呢？
      1.1 该把数据准备好呢？准备成啥样啥格式呢？
      1.2 批次是不是不用考虑？一次加载进去，然后tf/keras自己靠batch参数控制？keras不是有批次输入么？
    2.该如何中途保存呢？
    3.该如何输出呢？
    4.如何验证正确率和loss？loss是错误率么？是一个batch评价一次么？
    其他问题：
      - 去研究一下minst的数据加载，如何填充(？, 1, image_height, image_width)第一个维度的？
          应该是一次都加载进去么？


    样本数据，有几个特点，所有的他的可能组合是0-9,a-z,A-Z，合计是62个，不包含大写的有36个
    切割方法由于存在粘连，识别起来比较费劲，所以，这个模型里面采用的直接识别，
    当然后续也可以去切割出来，即使粘连，可能还是可以单个识别的吧，但是后面再做尝试了。

    那么粘连在一起的时候，识别的就变成了一个多分类问题，而不是单分类问题了，
    也就是说，我几个数字一起识别，那么就有以下几个问题：
    1. 怎么识别多个字母？
        这里的解决办法，是靠经验判断大多数是几个字母，
        比如大多数是4位的，极少数是5个，那么就放弃5个的，直接认为是4分类。
        但是我认为如果超过5%的少数，就得按照少数来吧。
        那么当定义为5位的时候，那么4位的缺失就用个padding代替吧，比如"_",程序自行判断是个padding吧

    2. 对于多分类，label的vector如何构建？
        对于单分类，我们都是知道是构建一个one-hot的概率向量，然后和结果做交叉熵
        对于多分类，就构建一个one-hot概率向量组成的张量呗，然后继续交叉熵

    3. 交叉熵函数如何书写？      
        好问题，还不知道呢，一会儿研究
'''

def train(image_width=75,image_height=32):

    letters = list('0123456789abcdefghijklmnopqrstuvwxyz')
    weight_decay = 0.001
    data_dir = "data"
    # batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。所以要适当选择
    batch_size = 50
    # 完整迭代次数
    epochs = 10
    #识别字符的数量    
    num_symbol = 5
    #模型的保存文件名
    model_name = 'model/mycaptcha.h5'

    num_classes = len(letters)*num_symbol

    # 训练集
    x_train,y_train = image_process.load_all_image_by_dir('data/train/')

    if os.path.exists(model_name):
        num_model = cnn.load_model(model_name)
        logger.info("加载已经存在的训练模型%s",model_name)
    else:
        logger.info("创建识别字符长度的模型")
        input_shape = None 
        if K.image_data_format() == 'channels_first': 
            input_shape = (1,image_height,image_width)
        else:
            input_shape = (image_height,image_width,1)   #这个shape不包含第一个维度，也就是图片数量

        
        #定义寻找字符个数的CNN
        num_model = cnn.create_model(input_shape,num_classes) 

        #定义判断字符的CNN
        #char_model = cnn.create_model()

    #训练期间保存checkpoint，防止整个crash
    #checkpoint_path = "num-model-{epoch:02d}-{val_acc:.2f}.hdf5" #这个是对save_best_only=False的文件名
    checkpoint_path = "model/checkpoint.hdf5"
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

if __name__ == '__main__':
    train()