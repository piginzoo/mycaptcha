#-*- coding:utf-8 -*- 
# 设置默认的level为DEBUG
# 设置log的格式
import logging as logger
logger.basicConfig(
    level=logger.DEBUG,
    format="[%(levelname)s] %(message)s"
)