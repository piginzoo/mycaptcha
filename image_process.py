# coding: utf-8
'''
#-----------------------------------------------------------------------------------------
				
				此文件用来加载一张图片，并且进行预处理

	主要做了以下操作：
		1.通过阈值调整去掉左右留白
		2.通过阈值调整去掉干扰线
		3.把突变成黑底的，白字的，100x36的，干净输入
	操作:
		二值化，顶格，字符分割等
	依赖库及版本:
		openCV > 2.4.x, skimage >= 0.9.x
	author:
		piginzoo
	date:
		2018/2
#------------------------------------------------------------------------------------------
'''

import numpy as np
import cv2
import logging as logger
import os
import time as t
from skimage.measure import regionprops
from skimage import morphology
from skimage.morphology import label
from skimage import color
import math
from skimage import data,filters
from skimage.morphology import disk
import debug,crash_debug
from keras import backend as K
import label as label_process

def dump_array_detail(arr):
	logger.debug("Dump the array:")
	for row in arr:
		logger.debug(row)

def output_img(name,img):
	file_name = os.path.basename(name)
	#调试用，不用打开了，否则，20000张图片，会撑爆硬盘的
	#cv2.imwrite("out/"+file_name+'.jpg',img)

def preprocess_image(imgname,width=75,height=32):
	#print imgname
	#print imgname.split(".")[0].split("/")[1]
	file_name = imgname.split(".")[0].split("/")[1]

	
	#按照灰度读入
	img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
	if(img is None):
		logger.error("无法加载文件，为空：%s",imgname)
		return None
	logger.debug( "img shape:%r",img.shape)
	output_img(file_name+"1原图",img)

	#gao_img = filters.gaussian(img,sigma=0.4) #sigma=0.4
	#output_img(file_name+"2模糊",gao_img)
	#gao_img = filters.median(img,disk(1))
	#gao_img = filters.gaussian_filter(img,sigma=0.4) #sigma=0.4
	thresh = filters.threshold_otsu(img) #返回一个阈值
	t =(img <= thresh)*1.0 #根据阈值进行分割
	output_img(file_name+"3阈值图像",t*255)

	# 选取一个全局阈值，然后就把整幅图像分成了非黑即白的二值图像了
	# 这个函数有四个参数，
	#	第一个原图像，
	#	第二个进行分类的阈值，
	# 	第三个是高于（低于）阈值时赋予的新值，
	#	第四个是一个方法选择参数，常用的有： 
	# 		• cv2.THRESH_BINARY（黑白二值） 
	# 		• cv2.THRESH_BINARY_INV（黑白二值反转） 
	# 该函数有两个返回值，
	# 第一个retVal（得到的阈值值（在后面一个方法中会用到）），
	# 第二个就是阈值化后的图像。 
	# 值>127==>255, <127==>0
	# retval, t = cv2.threshold(gao_img, 127, 255, cv2.THRESH_BINARY_INV)

	# #===>把图像给二值化了，做了黑白反转，阈值是127，凡是低于阈值的都设为1
	# #debug了一下，就是把原来白的都变黑为0（靠THRESH_BINARY_INV这个参数），
	# #            然后原来黑的地方都变成1
	# logger.debug( "图像做了阈值二值化处理，大于127=>255, 小于127=>0")
	# output_img(file_name+"3二值处理",t)

	#删除掉小的区块，面积是minsize=25,25是个拍脑袋的经验值
	#要先转成bool数组，true表示1，false：0
	t = t > 0		
	#返回值也是个bool数组
	t = morphology.remove_small_objects(t,min_size=25,connectivity=1)

	output_img(file_name+"4删除小块",t*255)#*255是为了变成白色的用于显示

	t = t.astype(np.float32)
	#防止有的图像不是规定的widthxheight，有必要都规范化一下
	if t.shape[1] < width:
		t = np.concatenate((t, np.zeros((width, width - t.shape[1]), dtype='uint8')), axis=1)
	else:
		t = cv2.resize(t, (width, height))

	#output_img(file_name+"规范大小",t*255)#*255是为了变成白色的用于显示

	
	#同时把原来白色的地方变成1，黑的地方为0，这个是训练要求，有数字显示的地方是1，没有的是0
	t = t > 0

	#变成一个四维numpy.ndarray,为了和[图片个数？,image channel,height,width]
	#tensorflow和thenano的格式要求不一样，图像通道的位置反着，烦人，做一下处理
	if K.image_data_format() == 'channels_first':
		I = t.astype(np.float32).reshape((1, 1, height, width)) 
	else:
	 	I = t.astype(np.float32).reshape((1, height, width, 1)) 

	logger.debug( "图像数据:%r",I.shape)
	return I,file_name #返回一个图像的矩阵，和文件的名字，这个名字就是标签


#"data/"
#返回是个4维度的numpy.ndarray[20000,1,100,36]，
#20000是图片数
#1是图像通道，灰度的
#100x36图像
def load_all_image_by_dir(path,image_width=75,image_height=32):
	start = t.time()
  	data = []
  	label = []
  	file_list = os.listdir(path)
  	debug_count = 0

  	for file in file_list:
  		# debug_count+=1
  		# if debug_count>100: break #调试用，省的加载那么长时间图片

  		# print file
  		if (file.find(".jpg")==-1): continue

  		one_img,img_name = preprocess_image(path+file,image_width,image_height)
  		if (one_img is None): continue
  		#加入数组
  		data.append(one_img)
  		label.append(label_process.label(img_name))
  		

  	#把数组堆叠到一起形成一个[20000,100,36,1]的张量	
  	image_data = np.vstack(data)

  	label_data = np.vstack(label)
	
  	logger.info("images data loaded:%r",image_data.shape)
  	end = t.time()
  	logger.info("加载图像使用了%d秒...",(end-start))

  	return image_data,label_data

if __name__ == '__main__':

	
	#建行的验证码尺寸
	IMG_WIDTH = 75
	IMG_HEIGHT = 32
	#测试整个data目录下的图片处理
  	data,label = load_all_image_by_dir("data/",IMG_WIDTH,IMG_HEIGHT)
  	logger.debug("加载了图像：%r,标签：%r",data.shape,label.shape)
  	#测试单张图片，可以看out输出
	#preprocess_image("data/99uwf.jpg",IMG_WIDTH,IMG_HEIGHT)  	
  	