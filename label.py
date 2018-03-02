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
def label(file_name,max_len=5):
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

if __name__ == '__main__':
	print label("trai5")		    
	print label("trai5555")
	print label("tra5")
	print label("tra")		    		    
