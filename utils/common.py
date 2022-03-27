# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 14:36
# @Author  : Qinglong
# @File    : common.py.py
# @Description: In User Settings Edit

import logging
import datetime
import os
from utils.configures import args


def init_logger():
    """
    logger.debug('this is a logger debug message')
    logger.info('this is a logger info message')
    logger.warning('this is a logger warning message')
    logger.error('this is a logger error message')
    logger.critical('this is a logger critical message')
    """
    # 第一步:创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    now_date = datetime.datetime.now()
    now_date = now_date.strftime('%Y-%m-%d_%H-%M-%S')

    # 第二步:创建一个handler，用于写入日志文件
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    log_file = os.path.join(args.log_dir, str(now_date) + ".log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    )
    # 添加handler到logger中
    logger.addHandler(file_handler)

    # 第三步:创建一个handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(console_handler)

    return logger
