#-*- coding:utf-8 -*-  
#这个文件玩用的，
import numpy as np
import cv2
import logging as logger
import os
import time as t
# from keras import backend as K

def dump_array_detail(arr):
	logger.debug("Dump the array:")
	for row in arr:
		logger.debug(row)

def output_img(name,img):
	file_name = os.path.basename(name)
	#调试用，不用打开了，否则，20000张图片，会撑爆硬盘的
	cv2.imwrite("out/"+file_name+'.jpg',img)

#load_image("type5/type5_train_1.png")
def load_image(imgname):

	# logger.info("loading %s...",imgname)
	img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
	if(img is None):
		logger.error("无法加载文件，为空：%s",imgname)
		return None
	logger.debug( "img shape:")
	logger.debug(img.shape)
	logger.debug("文件名：%s，文件大小：%r",imgname,img.shape)


	return 
	#logger.debug( len(img))
	#for row in img:
	#	logger.debug( row)

	#cv2.imshow(imgname,img)
	#cv2.waitKey(0)

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
	retval, t = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
	#cv2.imshow("t",t)
	#cv2.waitKey(0)
	#===>把图像给二值化了，做了黑白反转，阈值是127，凡是低于阈值的都设为1
	#debug了一下，就是把原来白的都变黑为0（靠THRESH_BINARY_INV这个参数），
	#            然后原来黑的地方都变成1
	logger.debug( "做了阈值二值化处理的")
	logger.debug(t)
	logger.debug(t.shape)
	output_img("binary_inv_img",t)
	for row in t:
		logger.debug(row)

	s = t.sum(axis=0)#210x64 ==> 210，降维，但是汇总了
	logger.debug( "s.size %d",len(s) )
	logger.debug( "二值化图像按列sum - s(是个图像数组):%s",s)
	#median：中位数，nonzeros(a)返回数组a中值不为零的元素的下标
	y1 = (s > np.median(s) + 5).nonzero()[0][0]
	logger.debug( "s>np.median(s):")
	logger.debug(s>np.median(s))
	logger.debug( "(s > np.median(s) + 5)")
	logger.debug((s > np.median(s) + 5))
	logger.debug( "(s > np.median(s) + 5).nonzero()")
	logger.debug((s > np.median(s)+5).nonzero())
	logger.debug( "y1:")
	logger.debug(y1)

	#折腾这一圈，我理解是为了去掉横线，所以np.median(s) + 5的5，我理解就是线粗

	y2 = (s > np.median(s) + 5).nonzero()[0][-1]
	x1, x2 = 0, 36
	#img数组的分片很诡异，不是x1:y1,x2:y2，
	#而是这里写的x1:x2, y1:y2
	#所以，这里作者认为识别码的竖着的方向，也就是x方向是0，36，说白了就是竖着36个像素
	#而，x，y方向和我们的理解的是一致的，但是numpy表示是反的
	#numpy表示的图像，高度（y坐标）在前，宽度（x坐标）在后
	#所以这个img[x1:x2...]的x1,x2其实是y1,y2，就是图像的高度
	im = img[x1:x2, y1 - 2:y2 + 3]
	logger.debug( "im:%r",im.shape)
	logger.debug(im)
	logger.debug( "x1:y2,x1:y2: %d,%d,%d,%d",x1,y1,x2,y2)
	output_img(imgname+"_cut_out_img",im)
	#cv2.imshow("2 values",im)
	#cv2.waitKey(0)

	#再黑白颠倒一下
	retval, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
	im0 = im[x1:x2, 1:-1]
	logger.debug( "im0:%r",im0.shape)
	logger.debug(im0)

	output_img(imgname+"_cut_out_inv_img",im0)
	#cv2.imshow("im0",im0)
	#cv2.waitKey(0)

	logger.debug( "new im:%r",im.shape)
	logger.debug(im)
	output_img(imgname+"_width_100_img",im)

	I = im > 127 #得到一个是不是白的100x36的二维布尔矩阵，白的为true，黑的为false
	logger.debug( "I:%r",I.shape)
	logger.debug(I)
	#变成一个四维numpy.ndarray,为了和[图片个数？,image channel,height,width]
	#同时把原来白色的地方变成1，黑的地方为0，不知道为何这样做的目的？？？
	# if K.image_data_format() == 'channels_first':
	# 	I = I.astype(np.float32).reshape((1, 1, 36, 100)) 
	# else:
	# 	I = I.astype(np.float32).reshape((1, 36, 100, 1)) 

	I = I.astype(np.float32).reshape((1, 1, 36, 100)) 

	logger.debug( "I:%r",I.shape)
	logger.debug(type(I))
	# logger.debug(I)
	dump_array_detail(I[0][0])
	# logger.info("loaded one image %r",I.shape)
	return I

# def sort_it(s):
# 	print s[s.index("train_")+6:s.index(".png")]
# 	return int(s[s.index("train_")+6:s.index(".png")])

#"data/"
#返回是个4维度的numpy.ndarray[20000,1,100,36]，
#20000是图片数
#1是图像通道，灰度的
#100x36图像
def load_all_image_by_dir(path):
	start = t.time()
  	data = []
  	file_list = os.listdir(path)
  	debug_count = 0

  	for file in file_list:
  		# debug_count+=1
  		# if debug_count>100: break #调试用，省的加载那么长时间图片

  		# print file
  		if (file.find(".jpg")==-1): continue

  		one_img = load_image(path+file)
  		if (one_img is None): continue
  		#加入数组
  		data.append(one_img)
  		output_img(file+"_small",one_img[0][0])

  	#把数组堆叠到一起形成一个[20000,100,36,1]的张量	
  	#new_data = np.vstack(data)
	
	
  	logger.info("images data loaded:%r",new_data.shape)
  	end = t.time()
  	logger.info("加载图像使用了%d秒...",(end-start))

  	return new_data

if __name__ == '__main__':

	# 设置默认的level为DEBUG
	# 设置log的格式
	logger.basicConfig(
	    level=logger.DEBUG,
	    format="[%(levelname)s] %(message)s"
	)

  	load_all_image_by_dir("data/")
  	

