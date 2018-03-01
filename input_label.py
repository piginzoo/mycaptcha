# coding: utf-8
import os
import csv
import numpy as np
import logging as logger
'''
解析type5_train.csv并且得到对应的识别码和他的长度
'''
def load_label(path):
	filename = os.path.join(os.getcwd(), path)
	label = []
	if os.path.exists(filename):
		with open(filename, 'r') as f:
		  reader = csv.reader(f)
		  for item in reader:
		    #print item[1],len(item[1])#识别码
		    logger.debug("load label:%s=>%s=>%d",item[0],item[1],len(item[1]))
		    label.append(len(item[1]))
	return np.array(label)

if __name__ == '__main__':
	# 设置默认的level为DEBUG
	# 设置log的格式
	logger.basicConfig(
	    level=logger.DEBUG,
	    format="[%(levelname)s] %(message)s"
	)
	load_label("image/type5_train/type5_train.csv")		    
