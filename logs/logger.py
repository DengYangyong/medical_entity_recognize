#coding:utf-8
import logging,os
import sys
sys.path.append("../")
from params import params

config = params()
log_dir = os.path.dirname(os.path.abspath(__file__))

def get_logger(log_file):
    
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s")
    
    fh = logging.FileHandler(os.path.join(log_dir,log_file))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

def print_config(config_dic, logger):
    """
    Print configuration of the model
    """
    for k, v in config_dic.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))      


""" 日志文件 """
logger = get_logger(config.log_file)
if config.train:
    print_config(vars(config),logger)    