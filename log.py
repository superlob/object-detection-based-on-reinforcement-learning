import logging
import os

log_dir = './log'

class Logger:
    def __init__(self, log_level=logging.INFO, log_format='%(asctime)s - %(levelname)s\n%(message)s\n'):
        self.log_dir = log_dir
        self.log_level = log_level
        self.log_format = log_format

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        log_path = os.path.join(self.log_dir, 'logger.txt')
        logging.basicConfig(
            level=self.log_level,
            format=self.log_format,
            filename=log_path,
            filemode='w'  # 覆盖模式
        )

    def info_test(self,message):
        logging.info(message)

    def info_memory(self,message):
        for exp in message:
            s = f"action:{message[1]} reward:{message[3]} done:{message[4]}"

log = Logger()