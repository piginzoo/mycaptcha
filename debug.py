#-*- coding:utf-8 -*- 
# 设置默认的level为DEBUG
# 设置log的格式
import logging as logger
logger.basicConfig(
    level=logger.INFO,
    format="[%(levelname)s] %(message)s"
)