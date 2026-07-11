import psutil
import os
import argparse
import logging
from datetime import datetime

def get_memory_usage():
    """
    Get the memory usage of the current process
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def get_memory_usage_of_all_processes():
    """
    Get the memory usage of all processes
    """
    return psutil.virtual_memory().percent


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def list_of_str(arg):
    return arg.split(",")

def str2bool(v):

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




class VerbosePrinter:
    def __init__(self, logger:logging.Logger=None):
        self.logger = logger
    def __call__(self, message:str, type:str="info"):
        if self.logger is not None:
            if type == "info":
                self.logger.info(message)
            elif type == "error":
                self.logger.error(message)
            elif type == "warning":
                self.logger.warning(message)
            else:
                raise ValueError(f"Invalid type: {type}")
        else:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{current_time} - [{type.upper()}] - {message}")

def verbose_print(message:str, type:str="info", logger:logging.Logger=None):

    if logger is not None:
        if type == "info":
            logger.info(message)
        elif type == "error":
            logger.error(message)
        elif type == "warning":
            logger.warning(message)
        else:
            raise ValueError(f"Invalid type: {type}")
    else:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - [{type.upper()}] - {message}")


def setup_logger(name:str=None, stream:bool=True, log_dir:str=None):

    """Set up logger configuration"""
    # Create logs directory in the parent directory of utils
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
    else:
        pass
    
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if name is None:
        log_file = os.path.join(log_dir, f'log_{timestamp}.log')
    else:
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    if stream:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    if stream:
        logger.addHandler(console_handler)
    
    return logger