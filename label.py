# coding: utf-8
'''
#-----------------------------------------------------------------------------------------
				
		解析标签数据

	主要做了以下操作：
		由于文件名就是标签，所以这个标签就是解析文件名
	author:
		piginzoo
	date:
		2018/2
#------------------------------------------------------------------------------------------
'''
import numpy as np
import logging as logger
import debug
from keras.utils import np_utils 

letters = list('0123456789abcdefghijklmnopqrstuvwxyz')

#入参是一个文件名的字符串，如 uv24c
#输出是一个62x5的一个5个one-hot向量组成的向量
def label2vector(file_name,max_len=5):
	length = len(file_name)
	
	char_index=[]
	for c in file_name:
		i = letters.index(c)
		char_index.append(i)

	if length > max_len:
		logger.debug("长度长了")
		char_index = char_index[:max_len]

	one_hots = np_utils.to_categorical(char_index,len(letters))

	if length< max_len:
		for i in range(max_len - length):
			one_hots = np.vstack(
				(one_hots,
				np.zeros(len(letters))))

	# print one_hots
	result = np.hstack(one_hots)
	# print result
	logger.debug( "标签数据:%r",result.shape)
	return result


#入参是一个62x5的一个5个one-hot向量组成的向量
#输出参是一个文件名的字符串，如 uv24c
def vector2label(vector,max_len=5):
	logger.debug("输入的hot向量是%r",vector)
	length = len(letters)
	assert vector.shape == (max_len*length,)

	#把一维62x5=180的向量，reshape成
	devided_vectors = np.reshape(vector,(-1,length))

	result = ""
	#Keras的to_categorical反向方法，就把1-hot变成一个数，就是字符串里的位置
	for one_hot in devided_vectors:
		logger.debug("当前的one-hot向量：%r",one_hot)
		#全0向量就忽略
		if np.count_nonzero(one_hot)==0:
			logger.debug("此向量为全0，解析为空！！！")
			continue
			
		index = np.argmax(one_hot)	
		letter = letters[index]
		logger.debug("解析字符的序号：%d,解析的字符为：%s",index,letter)
		result+= letter

	logger.debug( "解析出来的结果为:%r",result)
	return result


if __name__ == '__main__':
	
	a = label2vector("trai5")		    
	b = label2vector("trai5555")
	c = label2vector("tra5")
	d = label2vector("tra")		    		    
	print a
	print b
	print c
	print d

	print vector2label(a)
	print vector2label(b)
	print vector2label(c)
	print vector2label(d)