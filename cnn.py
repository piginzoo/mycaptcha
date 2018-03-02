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
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

DROPOUT_RATE = 0.25

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
	model.compile( 
			optimizer=keras.optimizers.Adadelta(),
            loss='binary_crossentropy',
            metrics=['accuracy'])

	return model 


if __name__ == '__main__':
	# batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。所以要适当选择
	batch_size = 128
	# 0-9手写数字一个有10个类别
	num_classes = 10
	# 12次完整迭代，差不多够了
	epochs = 4
	# 输入的图片是28*28像素的灰度图
	img_rows, img_cols = 28, 28
	# 训练集，测试集收集非常方便
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	 
	# keras输入数据有两种格式，一种是通道数放在前面，一种是通道数放在后面，
	# 其实就是格式差别而已
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	# 把数据变成float32更精确
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	# 把类别0-9变成2进制，方便训练
	y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)


	model = create_model(input_shape,num_classes)

	# 令人兴奋的训练过程
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
	          verbose=1, validation_data=(x_test, y_test))

	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])