# coding: utf-8
'''
#-----------------------------------------------------------------------------------------
                
                定义一个神经网络，标准的CNN

    主要做了以下操作：
       data()
            Conv-Layer() 

        openCV > 2.4.x, skimage >= 0.9.x
    author:
        piginzoo
    date:
        2018/2
#------------------------------------------------------------------------------------------
'''  
from __future__ import print_function
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import logging as logger
from keras import utils 


DROPOUT_RATE = 0.25
letters = list('0123456789abcdefghijklmnopqrstuvwxyz')

#此函数很重要。别看短。
#他是用来告诉tensorflow，你预测出来的和我给的label是否一致，
#tf预测出来的是一个杂乱无章的180维向量，而label是一个onehot的，
#例如：预测是[0.02,0.031,..,0.35,..,0.12]，label是[0,0..,1,..,0]
#那么我们处理方式如下，先把预测变成one-hot的，使用argmax+to_categorical,
#但是这个是在tf的session里面跑的，无法直接使用
#会报类似于的错： 
#ValueError: No default session is registered. 
#Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`
#所以，解决办法就是使用Keras的backend，也就是tf的张量函数来处理
#最后，把y_pred转成onehots后,在用backend的k_all比较得出结论
def custom_accuracy(y_true, y_pred):

	length = len(letters)
	print (y_pred)
	print (y_pred.shape)
	print (y_pred.shape[1].value)
	print (type(y_pred.shape[1].value))
	batch_size = y_pred.shape[1].value
	logger.debug("Batch size:%d",batch_size)

	tf.Print(y_true,[y_true],"y_true",summarize=20,first_n=5)
	tf.Print(y_pred,[y_pred],"y_pred",summarize=20,first_n=5)

	y_pred_reshape = K.reshape(y_pred,(-1,length))
	indexs = K.argmax(y_pred_reshape,axis=0)
	# one_hots = utils.to_categorical(indexs,length)
	_index = tf.expand_dims(indexs,1)
	sequence = tf.expand_dims(tf.range(0, length,dtype=tf.int64),1)#我觉得sparse_to_dense要这个变量有屁用啊？！
	concated = tf.concat(values=[sequence,_index],axis=1)
	one_hots = tf.sparse_to_dense(
		sparse_indices=concated, 
		output_shape=(batch_size,length), 
		sparse_values=1, 
		default_value=0)

	y_pred_one_hots = tf.concat(one_hots,axis=1)

	logger.debug("TF内部评估的标签是：%r",y_true)
	logger.debug("TF内部评估的预测是：%r",y_pred_one_hots)

	#return K.mean(K.equal(y_pred,y_true))

	return K.mean(K.equal(y_pred_one_hots,
	 	tf.cast(y_true,tf.int32)))


#input_shape，主要是确认
def create_model(input_shape,num_classes):
	# 牛逼的Sequential类可以让我们灵活地插入不同的神经网络层
	model = Sequential()
	# 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
	# 卷积核的窗口选用3*3像素窗口
	model.add(Conv2D(32,(3,3),activation='relu',strides=1,input_shape=input_shape))
	# 池化层是2*2像素的
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# 对于池化层的输出，采用DROPOUT_RATE概率的Dropout
	model.add(Dropout(DROPOUT_RATE))

	# 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
	# 卷积核的窗口选用3*3像素窗口
	model.add(Conv2D(32,(3,3),activation='relu',strides=1,input_shape=input_shape))
	# 池化层是2*2像素的
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# 对于池化层的输出，采用DROPOUT_RATE概率的Dropout
	model.add(Dropout(DROPOUT_RATE))

	# 展平所有像素，比如[36*100] -> [3600]
	model.add(Flatten())
	# 对所有像素使用全连接层，输出为128，激活函数选用relu
	model.add(Dense(1024, activation='relu'))
	# 对输入采用0.5概率的Dropout
	model.add(Dropout(DROPOUT_RATE))


	# 模型我们使用交叉熵损失函数，最优化方法选用Adadelta
	# 这个注释掉了，categorical_crossentropy适合softmax，多分类选1个，不适合我们的场景
	#model.compile(loss=keras.metrics.categorical_crossentropy,
	#              optimizer=keras.optimizers.Adadelta(),
	#              metrics=['accuracy'])
	#改成binary_crossentropy，用于多分类选多个的场景
	#但是之前要加上一个sigmod层，参见例子：https://keras.io/getting-started/sequential-model-guide/#training
	model.add(Dense(num_classes, activation='sigmoid'))

    #此处有问题，我评测的时候，y是我的验证结果集，而模型跑出来的是一个180维度向量
    #它怎么就判断这个模型结果和y就是一样的呢？
    #我可以自己通过找出180 reshape成5个向量后，寻找每个向量里面最大的那个下标确定的结果的
    #也就是说，我是通过外面的代码自己实现的，模型内部肯定是做不了这个的
    #除非我给他传入一个正确判断的函数进去

	model.compile( 
			optimizer=keras.optimizers.Adadelta(),
            loss='binary_crossentropy',
            metrics=['accuracy',custom_accuracy])

	return model 


if __name__ == '__main__':
	global letters
	letters = "123"
	import numpy as np
	a_true = np.array([[0,1,0],[0,1,0]])
	b_true = np.array([[0.5,1.2,0.3],[0.51,1.21,0.13]])
	print (custom_accuracy(a_true,b_true))#should be true
	a_true = np.array([[0,1,0]])
	b_true = np.array([[1.5,0.2,0.3]])
	print (custom_accuracy(a_true,b_true))#should be false
	